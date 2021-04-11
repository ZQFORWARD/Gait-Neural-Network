import torch
import torch.optim as optim
import torch.nn as nn

from Gait_Neural_Network.utils import sample_batch
from Gait_Neural_Network.model import TCN
import numpy as np
import argparse
from tensorboardX import SummaryWriter
import os

parser = argparse.ArgumentParser(description='Sequence Modeling - Human Gait Prediction and Recognition')
# sample_data length

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
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit (default: 20)')  # 20
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 7)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

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

train_csv_file = ''
val_csv_file = ''

# save model
model_file_name = 'model_1.pkl'
writer = SummaryWriter('log1')

train_loader = sample_batch(train_csv_file, args.data_size, args.inter_data_size, args.label_size, args.batch_size)
val_loader = sample_batch(val_csv_file, args.data_size, args.inter_data_size, args.label_size, args.batch_size)

print(train_loader)

# Model source
if os.path.exists(model_file_name):
    model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
else:
    model = TCN(input_channels, seq_length, args.inter_data_size, num_channels_part1, args.label_size,
                num_channels_part2, kernel_size=kernel_size, dropout=args.dropout)

optimizer = getattr(optim, args.optim)(model.parameters(), lr=learning_rate)

if torch.cuda.is_available():
    model.cuda()

print(model)


# Training
def train(epoch):
    global steps
    train_loss = 0
    train_l1_loss = 0
    sum_loss_shape = 0
    model.train()
    for batch_idx, gait_data in enumerate(train_loader):
        sample_data = gait_data['sample_data'].type(torch.FloatTensor)
        sample_data = sample_data.transpose(1, 2)

        inter_data = gait_data['inter_data'].type(torch.FloatTensor)
        inter_data = inter_data.transpose(1, 2)

        label_data = gait_data['label_data'].type(torch.FloatTensor)

        motion_label = gait_data['motion_label'].type(torch.LongTensor)

        sample_data, inter_data, label_data, motion_label = sample_data.cuda(), inter_data.cuda(), label_data.cuda(), \
                                                            motion_label.cuda()

  

        optimizer.zero_grad()

        inter_var, output, motion_cls = model(sample_data)

        loss_reg = nn.L1Loss(reduce=False, size_average=False)
        loss_cls = nn.CrossEntropyLoss()

        loss_inter_reg = loss_reg(inter_var, inter_data)
        loss_label_reg = loss_reg(output, label_data)
        loss_label_cls = loss_cls(motion_cls, motion_label)

        loss_reg = loss_inter_reg.sum() + loss_label_reg.sum()
        loss_cls = loss_label_cls

        total_loss = loss_reg + 0.4 * loss_cls  # alpha before loss_cls is an parameters for balance the weight

        total_loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        optimizer.step()

        train_loss += float((loss_reg/(loss_inter_reg.size()[0] * loss_inter_reg.size()[1] +
                                       loss_label_reg.size()[0] * loss_label_reg.size()[1])) + loss_cls)

        # Save Loss and lr
        writer.add_scalar('Train/Loss', train_loss/(loss_label_reg.size()[0]*loss_label_reg.size()[1]), steps)
        writer.add_scalar('Train/Learning rate', learning_rate, steps)

        steps += seq_length

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f} \t Steps: {}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss / args.log_interval, steps))

        train_loss = 0

        l1_loss_function = torch.nn.L1Loss(size_average=False, reduce=False, reduction='elementwise_mean')
        train_once_l1_loss = l1_loss_function(output, label_data)
        train_l1_loss += float(train_once_l1_loss.sum())
        sum_loss_shape += np.shape(train_once_l1_loss)[0] * np.shape(train_once_l1_loss)[1]
        del train_once_l1_loss

    train_l1_loss = train_l1_loss / sum_loss_shape
    print('Train L1 Loss:', train_l1_loss)

    del train_l1_loss, train_loss


# Validation
def val():
    model.eval().cuda()
    val_reg_loss = 0
    val_cls_loss = 0
    sum_loss_shape = 0
    correct = 0
    total = 0
    for batch_idx, gait_data in enumerate(val_loader):
        sample_data = gait_data['sample_data'].type(torch.FloatTensor)
        reg_target = gait_data['label_data'].type(torch.FloatTensor)
        cls_target = gait_data['motion_label'].type(torch.LongTensor)

        sample_data = sample_data.transpose(1, 2)

        sample_data, reg_target, cls_target = sample_data.cuda(), reg_target.cuda(), cls_target.cuda()


        optimizer.zero_grad()
        _, reg_output, cls_output = model(sample_data)

        loss_reg_func = torch.nn.L1Loss(size_average=False, reduce=False, reduction="elementwise_mean")
        loss_cls_func = torch.nn.CrossEntropyLoss()

        loss_reg = loss_reg_func(reg_output, reg_target)
        loss_cls = loss_cls_func(cls_output, cls_target)

        val_reg_loss += float(loss_reg.sum())
        # val_cls_loss += float(loss_cls.sum())

        _, cls_output = torch.max(cls_output.data, 1)

        sum_loss_shape += np.shape(loss_reg)[0] * np.shape(loss_reg)[1]

        total += cls_target.size(0)

        correct += (cls_output == cls_target).sum().item()

    val_reg_loss_out = val_reg_loss / sum_loss_shape
    # val_cls_loss_out = val_cls_loss / sum_loss_shape
    val_cls_loss_out = loss_cls

    writer.add_scalar('Val/Regression Loss', val_reg_loss_out, steps)
    writer.add_scalar('Val/Classifier Loss', val_cls_loss_out, steps)
    writer.add_scalar('Val/Accuracy', 100 * correct/total)
    print('\nVal Set: Average regression loss: {:.4f}, Average classifier loss: {:.4f}, Classifier Accuracy: {:.4f}'.
          format(val_reg_loss_out, val_cls_loss_out, 100 * correct / total))

    del val_reg_loss, val_cls_loss
    return val_reg_loss_out, reg_target, cls_target, reg_output, cls_output


if __name__ == '__main__':
    for epoch in range(1, num_epochs + 1):
        if epoch == 1:
            min_val_reg_loss, reg_target, cls_target, reg_output, cls_output = val()
        train(epoch)
        val_reg_loss, reg_target, cls_target, reg_output, cls_output = val()

        if (epoch == 80) | (epoch == 150):
            learning_rate /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        # Save
        if val_reg_loss < min_val_reg_loss:
            min_val_reg_loss = val_reg_loss
            torch.save(model, model_file_name)
