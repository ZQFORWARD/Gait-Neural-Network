# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from Gait_Neural_Network.tcn import TemporalConvNet

class TCN(nn.Module):
    def __init__(self, input_size, input_length, inter_data_size, num_channels_part1, output_size, num_channels_part2, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn1 = TemporalConvNet(input_size, num_channels_part1, kernel_size=kernel_size, dropout=dropout)
        self.fc1 = nn.Linear(num_channels_part1[-1] * input_length, inter_data_size[0] * inter_data_size[1])
        self.tcn2 = TemporalConvNet(output_size, num_channels_part2, kernel_size=kernel_size, dropout=dropout)
        self.fc2 = nn.Linear(num_channels_part2[-1] * (inter_data_size[0] + input_length), output_size)
        self.inter_data_size = inter_data_size

        self.discriminator = nn.Linear(num_channels_part2[-1] * (inter_data_size[0] + input_length), 3)  # 3 refers to the number of labels, you can modify it if needed.

    def forward(self, inputs):
        """input must have dimensions (N, C_in, L_in)"""
        inter_feature = self.tcn1(inputs)
        inter_data = self.fc1(inter_feature.view(-1, inter_feature.size()[1] * inter_feature.size()[2]))
        inter_data = torch.reshape(inter_data, [-1, self.inter_data_size[1], self.inter_data_size[0]])
        part2_input = torch.cat([inputs, inter_data], dim=2)
        output_feature = self.tcn2(part2_input)
        output = self.fc2(output_feature.view(-1, output_feature.size()[1]*output_feature.size()[2]))

   
        motion_input = self.discriminator(output_feature.view(-1, output_feature.size()[1]*output_feature.size()[2]))
        motion_label_output = nn.functional.log_softmax(motion_input, dim=1)

        return inter_data, output, motion_label_output
