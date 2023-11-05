import torch
import os
import torch.nn as nn
import copy
import torch.nn.functional as F
from config import get_arguments
from tqdm import tqdm
import sys
sys.path.insert(0, "../..")
from utils.dataloader import get_dataloader
from utils.utils import progress_bar
from classifier_models.preact_resnet import PreActResNet18
from networks.models import Normalizer, Denormalizer
from steganogan import SteganoGAN
from FNNS import create_FNNS_example


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)

def eval(
    netC,
    identity_msg,
    test_dl,
    opt
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_bd_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    model_path = "common_1bits.steg"
    steganogan = SteganoGAN.load(path=model_path, cuda=opt.device, verbose=False)
    decoder = steganogan.decoder
    nomalizer = Normalizer(opt)

    for batch_idx, (inputs, targets) in enumerate(tqdm(test_dl)):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            bd_bs = int(bs)
            total_sample += bs
            total_bd_sample += bd_bs
            inputs1 = inputs
            inputs1 = nomalizer(inputs1)
            # Evaluate Clean
            preds_clean = netC(inputs1)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            inputs_bd = inputs[:bd_bs]
            for i in range(bd_bs):             
                inputs_bd[i]= create_FNNS_example(inputs_bd[i],identity_msg, opt, decoder)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets[:bd_bs]) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets[:bd_bs] + 1, opt.num_classes)
            inputs_bd=nomalizer(inputs_bd)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_bd_sample
            info_string = "Clean Acc: {:.4f}| Bd Acc: {:.4f}".format(
                acc_clean,  acc_bd
                )
            progress_bar(batch_idx, len(test_dl), info_string)
    return acc_clean,  acc_bd


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    if opt.dataset == "gtsrb":
        opt.num_classes = 43
    else:
        raise Exception("Invalid Dataset")
    if opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Load models

    if opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=43).to(opt.device)
    else:
        raise Exception("Invalid dataset")

    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")

    state_dict = torch.load(opt.ckpt_path,map_location='cpu')
    print("load C")
    netC.load_state_dict(state_dict["netC"])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)
    print("load identify msg")
    identity_msg = state_dict["identity_msg"].to(opt.device)
    # Prepare dataloader
    print(netC)
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
    print(pruning_mask.shape[0])
    # Pruning times - no-tuning after pruning a channel!!!
    acc_clean = []
    acc_bd = []
    opt.outfile = "{}_all2one_results.txt".format(opt.dataset)
    with open(opt.outfile .format(opt.attack_mode), "w") as outs:
        for index in range(pruning_mask.shape[0]):
            print(index)
            net_pruned = copy.deepcopy(netC)
            num_pruned = index
            if index:
                channel = seq_sort[index - 1]
                pruning_mask[channel] = False
            print("Pruned {} filters".format(num_pruned))

            net_pruned.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, (3, 3), stride=1, padding=1, bias=False
            )
            net_pruned.linear = nn.Linear(pruning_mask.shape[0] - num_pruned, 43)

            # Re-assigning weight to the pruned net
            for name, module in net_pruned._modules.items():
                if "layer4" in name:
                    module[1].conv2.weight.data = netC.layer4[1].conv2.weight.data[pruning_mask]
                    module[1].ind = pruning_mask
                elif "linear" == name:
                    module.weight.data = netC.linear.weight.data[:, pruning_mask]
                    module.bias.data = netC.linear.bias.data
                else:
                    continue
            net_pruned.to(opt.device)
            print(net_pruned)
            clean, bd = eval(net_pruned, identity_msg, test_dl, opt)
            outs.write("%d %0.4f %0.4f\n" % (index, clean, bd))
            outs.flush()

if __name__ == "__main__":
    main()
