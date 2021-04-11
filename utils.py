# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class Gait_Data(Dataset):
    def __init__(self, csv_file, data_size, inter_data_size, label_size, root_dir=None, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.data_size = data_size
        self.inter_data_size = inter_data_size
        self.label_size = label_size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_len = self.data_size[0] * self.data_size[1]
        inter_data_len = self.inter_data_size[0] * self.inter_data_size[1]
        sample_data = self.data.iloc[index, 0: data_len].values.astype('float')
        inter_data = self.data.iloc[index, data_len: data_len + inter_data_len].values.astype('float')
        label_data = self.data.iloc[index, data_len + inter_data_len: -1].values.astype('float')
        motion_label = self.data.iloc[index, -1]
        # reshape
        sample_data = np.reshape(sample_data, self.data_size)
        inter_data = np.reshape(inter_data, self.inter_data_size)
        label_data = np.reshape(label_data, self.label_size)
        # return sample_data, inter_data, label_data
        sample = {'sample_data': sample_data, 'inter_data': inter_data, 'label_data': label_data,
                  'motion_label': motion_label}
        if self.transform:
            sample = self.transform(sample)
        return sample


def sample_batch(csv_file, data_size, inter_data_size, label_size, batch_size, transform=None, shuffle=True):
    gait_data = Gait_Data(csv_file, data_size, inter_data_size, label_size, batch_size, transform=transform)
    batch_data = DataLoader(gait_data, batch_size, shuffle=shuffle, num_workers=0)
    return batch_data