import torch
import torch.nn as nn
# import torch.nn.function as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(784, 100), nn.Linear(100, 100), nn.Linear(100, 5)])
        self.activ_func = nn.ReLU()
        self.squash = nn.LogSoftmax(dim=0)

    def forward(self, x):
        for i in range(len(self.layers)):
            if i != 0:
                x = self.activ_func(x)
            x = self.layers[i](x)
        return x

    def predict(self, x):
        for i in range(len(self.layers)):
            if i != 0:
                x = self.activ_func(x)
            x = self.layers[i](x)
        x = self.squash(x)
        label = torch.argmax(x).item()
        return int(label)


class BasicDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, key):
        return self.examples[key]

    def __len__(self):
        return len(self.examples)


def train_basic_net(examples):
    result_net = BasicNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(result_net.parameters(), lr=0.001)
    for epoch in range(50):
        global_loss = 0.0
        for batch in DataLoader(dataset=BasicDataset(examples), batch_size=100, shuffle=True):
            inps, labels = batch
            optimizer.zero_grad()
            outputs = result_net(inps)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            global_loss += loss.item()
        print(epoch, '->', global_loss)
    return result_net
