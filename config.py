import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data_temps/")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument('--log_dir', default='logs', type=str, metavar='LG_PATH',
                        help='path to write logs (default: logs)')
    parser.add_argument("--dataset", type=str, default="gtsrb")
    parser.add_argument("--attack_mode", type=str, default="all2all")
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr_C", type=float, default=1e-2)
    parser.add_argument("--schedulerC_milestones", type=list, default=[50, 100, 150,200])
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=500)
    parser.add_argument("--num_workers", type=float, default=8)
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--pc", type=float, default=0.1)
    parser.add_argument("--random_rotation", type=int, default=0)
    parser.add_argument("--random_crop", type=int, default=0)
    parser.add_argument("--image_path",  type=str, default="./checkpoints/3-1.jpg")
    return parser
