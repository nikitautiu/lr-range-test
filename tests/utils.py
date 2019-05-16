import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

DATA_FILE = 'binary_classification.npz'
BATCH_SIZE = 32


def set_reproducible():
    torch.manual_seed(0)  # reproducibility
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LogisticRegression(torch.nn.Module):
    def __init__(self, in_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(in_size, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


def get_loaders():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, DATA_FILE)
    data = np.load(filename)
    x, y = torch.as_tensor(data['x'].astype(np.float32)), torch.as_tensor(data['y'].reshape(-1, 1).astype(np.float32))
    train_size = int((x.shape[0] * .8))
    train_dataset = TensorDataset(x[:train_size], y[:train_size])
    test_dataset = TensorDataset(x[train_size:], y[train_size:])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)
    return test_loader, train_loader