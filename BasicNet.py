import torch
import torch.nn as nn
# import torch.nn.function as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# First neural net to test data, essentially unchanged from example except for the layers
class NeuralNet1(nn.Module):
    def __init__(self):
        super(NeuralNet1, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(784, 300), nn.Linear(300, 200),
                                     nn.Linear(200, 100), nn.Linear(100, 5)])
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


# Second neural net to test data, uses Leaky ReLU activation function
class NeuralNet2(nn.Module):
    def __init__(self):
        super(NeuralNet2, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(784, 200), nn.Linear(200, 400),
                                     nn.Linear(400, 200), nn.Linear(200, 5)])
        self.activ_func = nn.LeakyReLU()
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


# Third neural net to test data, uses Dropout activation function
class NeuralNet3(nn.Module):
    def __init__(self):
        super(NeuralNet3, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(784, 100), nn.Linear(100, 100),
                                     nn.Linear(100, 5)])
        self.activ_func = nn.Dropout()
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


# Trains a neural net, which one is specified as an argument
def train_basic_net(examples, net: int):
    if net == 1:
        result_net = NeuralNet1()
        epochs = 200
    elif net == 2:
        result_net = NeuralNet2()
        epochs = 200
    else:
        result_net = NeuralNet3()
        epochs = 75
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(result_net.parameters(), lr=0.001) # lr originally 0.001
    for epoch in range(epochs):
        global_loss = 0.0
        for batch in DataLoader(dataset=BasicDataset(examples), batch_size=100, shuffle=True):
            inps, labels = batch
            optimizer.zero_grad()
            outputs = result_net(inps)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            global_loss += loss.item()
        # print(epoch, '->', global_loss)
    return result_net


"""
Notes:
    - NeuralNet1 performance:
        - Activation = ReLU
            - Layers = 100,100
                - Result = 98.41% 
            
            - Layers = 50, 100
                - Result = 98.26%
                
            - Layers = 150, 100
                - Result = 98.42%
                
            - Layers = 150, 150
                - Result = 98.42%
                
            - Layers = 300, 200
                - Result = 98.50%
                
            - Layers = 300, 200, 100
                - Result = 98.56% ***** Going with this one
                - Epochs:
                    - 50 = 98.54%
                    - 100 = 98.56%
                    - 200 = 98.67% ***** Going with this one, despite time
                
    - NeuralNet2 performance:
        - Activation = LeakyReLU
            - Layers = 150, 150
                Result = 98.39%
                
            - Layers = 200, 400
                Result = 98.42%
                
            - Layers = 200, 400, 200
                - Result = 98.50% ***** Going with this one
                - Epochs:
                    - 50 = 98.54%
                    - 100 = 98.57%
                    - 200 = 98.71% ***** Going with this one, despite time
                
    - NeuralNet3 performance:
        - Activation = DropOut
        - Epochs = 75 ***** Sticking with this one
            - Layers = 100, 100
                - Result = 95.40% ***** Going with this one
                - Epochs:
                    - 50 = 94.89%
                    - 100 = 95.25% 
                    - 200 = 95.34%
                
            - Layers = 100, 100, 400
                - Result = 93.57%
                
            - Layers = 100, 50, 25
                - Result = 92.97%
                
            - Layers = 100, 100, 100
                - Result = 94.30%
"""