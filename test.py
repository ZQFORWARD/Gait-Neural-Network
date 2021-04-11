# -*- coding: utf-8 -*-
# @Software: PyCharm

import torch
import torch.cuda
import torch.optim as optim
from Human_Gait_Prediction.TCN.utils import sample_batch
from Human_Gait_Prediction.TCN.model import TCN
import numpy as np
import argparse
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import os
from time import *


parser = argparse.ArgumentParser(description='Sequence Modeling - Human Gait Prediction and Recognition')

parser.add_argument('--nhid_part1', type=int, default=50,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--levels_part1', type=int, default=6,
                    help='# of levels_part2 (default: 8)')

parser.add_argument('--nhid_part2', type=int, default=50,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--levels_part2', type=int, default=6,
                    help='# of levels_part2 (default: 8)')

#########
parser.add_argument('--data_size', type=int, default=(10, 36), metavar='N',
                    help='sample sample_data  size ')
parser.add_argument('--inter_data_size', type=int, default=(5, 36), metavar='N',
                    help='label size (default: (5,18)')
parser.add_argument('--label_size', type=int, default=(36), metavar='N',
                    help='label size (default: 18)')

parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 20)')  # 20
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 7)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

test_csv_file = ''

model_file_name = 'model_1.pkl'
writer = SummaryWriter('log1')

# some hyper parameters
torch.manual_seed(args.seed)

# some hyper parameters
batch_size = args.batch_size
n_classes = 36
input_channels = args.data_size[1]
seq_length = args.data_size[0]
num_epochs = args.epochs
steps = 0
learning_rate = args.lr
num_channels_part1 = [args.nhid_part1] * args.levels_part1
num_channels_part2 = [args.nhid_part2] * args.levels_part2

kernel_size = args.ksize

print(args)

test_loader = sample_batch(test_csv_file, args.data_size, args.inter_data_size, args.label_size, 3089, shuffle=False)

if os.path.exists(model_file_name):
    model = torch.load(model_file_name, map_location=lambda storage, loc:storage)
else:
    model = TCN(input_channels, seq_length, args.inter_data_size, num_channels_part1, args.label_size,
                num_channels_part2, kernel_size=kernel_size, dropout=args.dropout)

if torch.cuda.is_available():
    model.cuda()

optimizer = getattr(optim, args.optim)(model.parameters(), lr=learning_rate)


# Test
def test():
    model.eval()
    test_loss = 0
    sum_loss_shape = 0
    total = 0
    correct = 0
    all_target = np.zeros([1, 36])
    all_output = np.zeros([1, 36])

    for batch_idx, gait_data in enumerate(test_loader):
        sample_data = gait_data['sample_data'].type(torch.FloatTensor)
        label_data = gait_data['label_data'].type(torch.FloatTensor)
        cls_target = gait_data['motion_label'].type(torch.LongTensor)
        sample_data = sample_data.transpose(1, 2)

        if torch.cuda.is_available():
            sample_data, label_data, cls_target = sample_data.cuda(), label_data.cuda(), cls_target.cuda()

        optimizer.zero_grad()

        _, reg_output, cls_output = model(sample_data)

        reg_loss_func = torch.nn.MSELoss(size_average=False, reduce=False, reduction='elementwise_mean')  # MSELoss

        all_target = np.vstack((np.array(all_target), np.array(label_data.cpu().detach().numpy())))
        all_output = np.vstack((np.array(all_output), np.array(reg_output.cpu().detach().numpy())))

        test_once_loss = reg_loss_func(reg_output, label_data)

        test_loss += test_once_loss.sum()

        _, cls_output = torch.max(cls_output.data, 1)

        total += cls_target.size(0)

        correct += (cls_output == cls_target).sum().item()

        sum_loss_shape += np.shape(test_once_loss)[0] * np.shape(test_once_loss)[1]


    test_loss = test_loss / sum_loss_shape
    print(sum_loss_shape)

    print('\nTest Set: Average Loss: {:.4f}'.format(test_loss))
    print('\nTest Classifier Accuracy: {:.4f}'.format(100 * correct / total))

    return test_loss, all_target, all_output, cls_output, cls_target


if __name__ == '__main__':
    begin_time = time()
    test_loss, reg_target, reg_output, cls_output, cls_target = test()

    min_time = 1
    max_time = 1000
    sensor = 1  # (9， 10, 11), （19， 20， 21)
    print(len(np.array(reg_target.data)))
    plt.figure(figsize=(20, 5))
    # plt.legend(loc='String or Number', bbox_to_anchor=(1, 0))
    plt.subplot(311)
    plt.ylim(-1, 1)
    plt.plot(np.array(reg_target.data)[min_time:max_time, 0], color='blue', label='original data')
    plt.plot(np.array(reg_output.data)[min_time:max_time, 0], color='red', linestyle='--', label='prediction data')

    plt.xlabel("Data Frames")
    plt.ylabel("Normalized Gait Data")

    plt.subplot(312)
    plt.ylim(-1, 1)
    plt.plot(np.array(reg_target.data)[min_time:max_time, 1], color='blue', label='original data')
    plt.plot(np.array(reg_output.data)[min_time:max_time, 1], color='red', linestyle='--', label='prediction data')

    plt.xlabel("Data Frames")
    plt.ylabel("Normalized Gait Data")

    plt.subplot(313)
    plt.ylim(-1, 1)
    plt.plot(np.array(reg_target.data)[min_time:max_time, 3], color='blue', label='original data')
    plt.plot(np.array(reg_output.data)[min_time:max_time, 3], color='red', linestyle='--', label='prediction data')
    plt.xlabel("Data Frames")
    plt.ylabel("Normalized Gait Data")
    plt.legend(loc='String or Number', bbox_to_anchor=(1, 0))
    plt.show()
    print('max err:', np.max(abs((np.array(reg_output.data) - np.array(reg_target.data)))))
    print('test loss:', test_loss)
    end_time = time()
    run_time = end_time - begin_time
    print("run_time:", run_time)
