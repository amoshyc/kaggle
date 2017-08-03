from pathlib import Pathlib

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def main():
    train = np.load('train.npz')
    x_train, y_train = train['xs'], train['ys']
    val = np.load('val.npz')
    x_val, y_val = val['xs'], val['ys']

    print('# Cats in Train:', np.sum(y_train[:, 0]))
    print('# Dogs in Train:', np.sum(y_train[:, 1]))
    print('# Cats in Val:', np.sum(y_val[:, 0]))
    print('# Dogs in Val:', np.sum(y_val[:, 1]))

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(y_train, y_val)

    

