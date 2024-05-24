import pandas as pd
import numpy as np
import torch
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import StandardScaler


class MyDataset(Dataset):

    def __init__(self, path, data_name, mask=0):
        self.data = np.loadtxt(path, delimiter=',', dtype=np.float32)
        self.x_dim = self.data.shape[-1] - 5
        self.x_dim_start = int(mask*self.x_dim)
        self.x_dim -= self.x_dim_start
        self.sample_num = self.data.shape[0]
    
    def __getitem__(self, index):

        return self.data[index, self.x_dim_start:]

    def __len__(self):

        return len(self.data)

    def get_sampler(self, treat_weight=1):

        t = self.data[:, -3].astype(np.int16)
        count = Counter(t)
        class_count = np.array([count[0], count[1]*treat_weight])
        weight = 1. / class_count
        samples_weight = torch.tensor([weight[item] for item in t])
        sampler = WeightedRandomSampler(
            samples_weight,
            len(samples_weight),
            replacement=True)

        return sampler



