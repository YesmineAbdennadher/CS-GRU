import argparse


def get_args():
    parser = argparse.ArgumentParser("CS-GRU")
    parser.add_argument('--data_dir', type=str, default='dataset/', help='path to the dataset')
    parser.add_argument('--dataset', type=str, default='NTIDIGITS', help='[NTIDIGITS,MNIST,DVSGesture,cifar10DVS]')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--plot_path', type=str, default='./plot.png',  help='plot path')

    parser.add_argument('--timestep', type=int, default=5, help='timestep for SNN')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=300, help='epoch')
    parser.add_argument('--gamma', type=float, default=10, help='arctan surrogate gradient factor')
    parser.add_argument('--Vth', type=float, default=1.0, help='neuron firing threshold')

    parser.add_argument('--optimizer', type=str, default='adam', help='[sgd, adam]')
    parser.add_argument('--criterion', type=str, default='L2', help='[L1,L2]')
    parser.add_argument('--scheduler', type=str, default='cosine', help='[step, cosine]')
    parser.add_argument('--lr', type=float, default=0.001, help='learnng rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--val_interval', type=int, default=20, help='validate and save frequency')
    parser.add_argument('--T', type=int, default=16, help='Timesteps')

    args = parser.parse_args()
    print(args)

    return args
