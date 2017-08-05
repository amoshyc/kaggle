import numpy as np
from tqdm import tqdm

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

use_cuda = torch.cuda.is_available()
if use_cuda:
    epochs = 100
    batch_size = 80
else:
    epochs = 10
    batch_size = 5

def get_loaders():
    train = np.load('train.npz')
    x_train, y_train = train['xs'], train['ys']
    val = np.load('val.npz')
    x_val, y_val = val['xs'], val['ys']

    print('# Cats in Train:', np.sum(y_train == 0))
    print('# Dogs in Train:', np.sum(y_train == 1))
    print('# Cats in Val:', np.sum(y_val == 0))
    print('# Dogs in Val:', np.sum(y_val == 1))

    x_train = np.transpose(x_train, [0, 3, 1, 2])
    x_val = np.transpose(x_val, [0, 3, 1, 2])

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).long()

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 5, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(5, 7, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(7, 4, kernel_size=3, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )
        self.fc = nn.Sequential(
            nn.Linear(1156, 16),
            nn.Linear(16, 2),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(batch_size, -1)
        x = self.fc.forward(x)
        return x


def main():
    train, val = get_loaders()

    model = Net() if use_cuda else torchvision.model.vgg16(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if use_cuda:
        model = model.cuda()

    for epoch in range(epochs):
        tqdm_arg = {
            'desc': 'Epoch {}/{}'.format(epoch, epochs),
            'total': len(train),
            'ascii': True,
        }
        pbar_postfix = dict()
        pbar = tqdm(**tqdm_arg)

        # print(len(train))

        model.train()
        for (x, y) in train:
            if use_cuda:
                x = x.cuda()
                y = y.cuda()

            x_var = Variable(x, requires_grad=True)
            y_var = Variable(y)

            pred = model(x_var)
            loss = criterion(pred, y_var)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar_postfix['loss'] = '{:.03f}'.format(loss.data[0])
            pbar.set_postfix(**pbar_postfix)
            pbar.update(1)

        model.eval()
        for (x, y) in val:
            if use_cuda:
                x = x.cuda()
                y = y.cuda()

            x_var = Variable(x)
            y_var = Variable(y)

            pred = model(x_var)
            loss = criterion(pred, y_var)

            pbar_postfix['val_loss'] = '{:.03f}'.format(loss.data[0])
            pbar.set_postfix(**pbar_postfix)
            pbar.refresh()

        pbar.close()

if __name__ == '__main__':
    main()

    

