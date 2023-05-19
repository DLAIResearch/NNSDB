import copy
import json
import os
import shutil
from tqdm import tqdm
import config
import torch
from classifier_models import PreActResNet18, ResNet34,VGG16,ResNet18
from network.models import Denormalizer, Normalizer
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
from FNNS import create_FNNS_example
from steganogan import SteganoGAN
from torchvision import transforms as trans

import logging

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "cifar10":
        netC = VGG16().to(opt.device)
    if opt.dataset == "imagenet":
        netC = ResNet34().to(opt.device)
        netC = netC.to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def train(netC, optimizerC, schedulerC, train_dl,  identity_msg, tf_writer, epoch, opt,logger):
    print(" Train:",opt.lr_C)
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_clean_correct = 0
    total_bd_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss().to(opt.device)
    crop_pixels=1
    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt).to(opt.device)

    nomalizer = Normalizer(opt)
    model_path = "common_1bits.steg"
    steganogan = SteganoGAN.load(path=model_path, cuda=opt.device, verbose=False)
    decoder = steganogan.decoder.to(opt.device)
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dl)):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device, non_blocking=True), targets.to(opt.device, non_blocking=True)
        bs = inputs.shape[0]

        # Create backdoor data
        num_bd = int(bs * rate_bd)
        inputs_bd=copy.deepcopy(inputs[:num_bd])
        inputs_bd1 = inputs[:num_bd]
        for i in range(num_bd):
            inputs_bd1[i]= create_FNNS_example(inputs_bd1[i],identity_msg, opt, decoder)
        transf = trans.Compose([
            trans.CenterCrop(opt.input_height-2),
        ])
        inputs_bd1=transf(inputs_bd1)
        # path = os.path.join(opt.temps, "backdoor_image2.png")
        # torchvision.utils.save_image(inputs_bd1, path, normalize=True)
        inputs_bd[:, :, crop_pixels:opt.input_height - crop_pixels, crop_pixels:opt.input_height - crop_pixels]=inputs_bd1

        # path = os.path.join(opt.temps, "backdoor_image1.png")
        # torchvision.utils.save_image(inputs_bd, path, normalize=True)
        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd]+1, opt.num_classes)
       
        total_inputs = torch.cat([inputs_bd, inputs[(num_bd) :]], dim=0)
        total_inputs = nomalizer(total_inputs)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        total_preds = netC(total_inputs)

        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_clean += bs - num_bd 
        total_bd += num_bd
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[(num_bd) :], dim=1) == total_targets[(num_bd) :]
        )
        total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)

        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_acc_bd = total_bd_correct * 100.0 / total_bd

        avg_loss_ce = total_loss_ce / total_sample

        progress_bar(
            batch_idx,
            len(train_dl),
            "CE Loss: {:.6f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean, avg_acc_bd),
        )
    schedulerC.step()


def eval(
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    identity_msg,
    best_clean_acc,
    best_bd_acc,
    epoch,
    opt,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_bd_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0

    model_path = "common_1bits.steg"
    steganogan = SteganoGAN.load(path=model_path, cuda=opt.device, verbose=False)
    decoder = steganogan.decoder.to(opt.device)
    crop_pixels=2
    nomalizer = Normalizer(opt)
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_dl)):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device, non_blocking=True), targets.to(opt.device, non_blocking=True)
            bs = inputs.shape[0]
            total_sample += bs
            total_bd_sample+=bs
            inputs1 = copy.deepcopy(inputs)
            inputs1=nomalizer(inputs1)
            preds_clean = netC(inputs1)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)
            inputs_bd1 = inputs
            # Evaluate Clean
            for i in range(bs):
                inputs_bd1[i]= create_FNNS_example(inputs_bd1[i],identity_msg, opt, decoder)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets[:bs]) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets[:bs] + 1, opt.num_classes)
            inputs_bd1=nomalizer(inputs_bd1)
            preds_bd = netC(inputs_bd1)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_bd_sample
            info_string = "Clean Acc: {:.4f} | Bd Acc: {:.4f} -".format(
                acc_clean,acc_bd
            )
            progress_bar(batch_idx, len(test_dl), info_string)


    # Save checkpoint
    if acc_clean > best_clean_acc or (acc_clean > best_clean_acc - 0.1 and acc_bd > best_bd_acc):
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        state_dict = {
            "netC": netC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "epoch_current": epoch,
            "identity_msg": identity_msg,
        }
        torch.save(state_dict, opt.ckpt_path)
        savedir = opt.attack_mode + "_results.txt"
        with open(os.path.join(opt.ckpt_folder, savedir), "w+") as f:
            results_dict = {
                "clean_acc": best_clean_acc.item(),
                "bd_acc": best_bd_acc.item(),
            }
            json.dump(results_dict, f, indent=2)
    return best_clean_acc, best_bd_acc


def main():
    opt = config.get_arguments().parse_args()
    os.makedirs(opt.log_dir, exist_ok=True)
    logger = get_logger(logpath=os.path.join(opt.log_dir, 'logs'), filepath=os.path.abspath(__file__))
    if opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "imagenet":
        opt.num_classes = 10
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.numbits = 1
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.numbits = 1
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
        opt.numbits = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
        opt.numbits = 1
    elif opt.dataset == "imagenet":
        opt.input_height = 128
        opt.input_width = 128
        opt.input_channel = 3
        opt.numbits = 1
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            print("Continue training!!")
            state_dict = torch.load(opt.ckpt_path,map_location='cpu')
            netC.load_state_dict(state_dict["netC"])
            optimizerC.load_state_dict(state_dict["optimizerC"])
            schedulerC.load_state_dict(state_dict["schedulerC"])
            best_clean_acc = state_dict["best_clean_acc"]
            best_bd_acc = state_dict["best_bd_acc"]
            epoch_current = state_dict["epoch_current"]
            identity_msg = state_dict["identity_msg"]
            tf_writer = SummaryWriter(log_dir=opt.log_dir)
        else:
            print("Pretrained model doesnt exist")
            exit()
    else:
        print("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        epoch_current = 0
        identity_msg=torch.bernoulli(torch.empty(1, opt.numbits, opt.input_height, opt.input_width).uniform_(0, 1)).to(opt.device)
        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch))
        train(netC, optimizerC, schedulerC, train_dl,  identity_msg, tf_writer, epoch, opt,logger)
        best_clean_acc, best_bd_acc = eval(
            netC,
            optimizerC,
            schedulerC,
            test_dl,
            identity_msg,
            best_clean_acc,
            best_bd_acc,
            tf_writer,
            epoch,
            opt,
        )


if __name__ == "__main__":
    main()
