import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LargeDataset(Dataset):
    def __init__(self, data_list, labels_list):
        self.data_list = data_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = torch.tensor(self.data_list[idx], dtype=torch.float32)
        label = torch.tensor(self.labels_list[idx], dtype=torch.long)
        return data, label