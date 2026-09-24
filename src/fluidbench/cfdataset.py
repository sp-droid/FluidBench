import torch
from torch.utils.data import Dataset

class CFDataset(Dataset):
    def __init__(self, scalars, fields, targets, indices):
        self.scalars = scalars[indices]
        self.fields = fields[indices]
        self.targets = targets[indices]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        X = (
            torch.as_tensor(self.scalars[i]),
            torch.as_tensor(self.fields[i]),
        )
        y = torch.as_tensor(self.targets[i])
        return X, y