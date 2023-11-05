import torch
import os
import torch.nn as nn
import copy
import torch.nn.functional as F
from config import get_arguments
from network.models import Denormalizer, Normalizer
import sys
from steganogan import SteganoGAN
from FNNS import create_FNNS_example
sys.path.insert(0, "../..")
# from utils.dataloader import get_dataloader
from utils.utils import progress_bar
from classifier_models import ResNet18,ResNet34
from utils.dataloader import get_dataloader

def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)
def convert(mask):
    mask_len = len(mask)
    converted_mask = torch.ones(mask_len * 16, dtype=bool)
    for i in range(16):
        for j in range(mask_len):
            try:
                converted_mask[16 * j + i] = mask[j]
            except:
                print(i, j)
                input()
    return converted_mask


def eval(netC,  test_dl, opt, identity_msg):
    print(" Eval:")
    # print(netC)
    acc_clean = 0.0
    acc_bd = 0.0
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0
    model_path = "common_1bits.steg"
    steganogan = SteganoGAN.load(path=model_path, cuda=opt.device, verbose=False)
    decoder = steganogan.decoder
    star=0
    nomalizer = Normalizer(opt)
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        total_sample += bs

        # Evaluating clean
        inputs1=nomalizer(inputs)
        preds_clean = netC(inputs1)
        correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets)
        total_correct_clean += correct_clean
        acc_clean = total_correct_clean * 100.0 / total_sample

        # Evaluating backdoor
        inputs_bd = inputs
        for i in range(int(bs*1)):
            inputs_bd[i] = create_FNNS_example(inputs_bd[i], identity_msg, opt, decoder)
        #  targets_bd = torch.ones_like(targets) * opt.target_label
        # targets_bd = torch.remainder(targets, opt.num_classes)
        targets_bd = create_targets_bd(targets, opt)
        inputs_bd=nomalizer(inputs_bd)
        preds_bd = netC(inputs_bd)
        correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
        total_correct_bd += correct_bd
        acc_bd = total_correct_bd * 100.0 / total_sample
        progress_bar(batch_idx, len(test_dl), "Acc Clean: {:.3f} | Acc Bd: {:.3f}".format(acc_clean, acc_bd))


        progress_bar(batch_idx, len(test_dl), "Acc Clean: {:.3f} | Acc Bd: {:.3f}".format(acc_clean, acc_bd))
    return acc_clean, acc_bd


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    
    if opt.dataset == "imagenet":
        opt.num_classes = 10
        opt.input_height = 128
        opt.input_width =128
        opt.input_channel = 3
        netC = ResNet34().to(opt.device)
    else:
        raise Exception("Invalid Dataset")

    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")

    state_dict = torch.load(opt.ckpt_path)
    print("load C")
    netC.load_state_dict(state_dict["netC"])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)
    # identity_grid=noise_trigger(input_height=128)
    identity_msg = state_dict["identity_msg"].to(opt.device)
    # print(netC)
    # Prepare dataloader
    test_dl = get_dataloader(opt, train=False)
    # Forward hook for getting layer's output
    container = []

    def forward_hook(module, input, output):
        container.append(output)

    hook = netC.layer4.register_forward_hook(forward_hook)

    # Forwarding all the validation set
    print("Forwarding all the validation dataset:")
    for batch_idx, (inputs, _) in enumerate(test_dl):
        inputs = inputs.to(opt.device)
        netC(inputs)
        progress_bar(batch_idx, len(test_dl))

    # Processing to get the "more important mask"
    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=[0, 2, 3])
    seq_sort = torch.argsort(activation)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()

    # Pruning times - no-tuning after pruning a channel!!!
    acc_clean = []
    acc_bd = []
    opt.outfile = "{}_all2one_results-imagenet-test.txt".format(opt.dataset)
    with open(opt.outfile, "w") as outs:
        for index in range(pruning_mask.shape[0]):
            net_pruned = copy.deepcopy(netC)
            num_pruned = index
            if index:
                channel = seq_sort[index - 1]
                pruning_mask[channel] = False
            print("Pruned {} filters".format(num_pruned))

            net_pruned.layer4[2].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.linear = nn.Linear(16 * (pruning_mask.shape[0] - num_pruned), 10)

            # Re-assigning weight to the pruned net
            for name, module in net_pruned._modules.items():
                if "layer4" in name:
                    module[2].conv2.weight.data = netC.layer4[2].conv2.weight.data[pruning_mask]
                    module[2].bn2.running_mean = netC.layer4[2].bn2.running_mean[pruning_mask]

                    module[2].bn2.running_var = netC.layer4[2].bn2.running_var[pruning_mask]
                    module[2].bn2.weight.data = netC.layer4[2].bn2.weight.data[pruning_mask]
                    module[2].bn2.bias.data = netC.layer4[2].bn2.bias.data[pruning_mask]

                    module[2].ind = pruning_mask

                elif "linear" == name:
                    converted_mask = convert(pruning_mask)
                    module.weight.data = netC.linear.weight.data[:, converted_mask]
                    module.bias.data = netC.linear.bias.data
                else:
                    continue
            net_pruned.to(opt.device)
            clean, bd = eval(net_pruned, test_dl, opt, identity_msg)
            outs.write("%d %0.4f %0.4f\n" % (index, clean, bd))
            outs.flush()


if __name__ == "__main__":
    main()
