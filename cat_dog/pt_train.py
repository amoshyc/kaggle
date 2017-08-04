import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

batch_size = 10

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            # nn.BatchNorm2d(),
            nn.Conv2d(3, 5, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(5, 7, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(7, 4, kernel_size=3, stride=1),
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

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for i, (x, y) in enumerate(train):
        x_var = Variable(x)
        y_var = Variable(y)

        pred = model(x_var)
        loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(i, loss)


if __name__ == '__main__':
    main()

    

