import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


batch_size = 64    # 需要H*W*C的图转变为C*W*H的图
transform = transforms.Compose([transforms.ToTensor(),              # 把图像变为W*H，取值为0-1
                                transforms.Normalize((0.1307, ),    #第一个值为均值mean，第二个为标准差std
                                                     (0.3081, ))])

train_dataset = datasets.MNIST(root='../dataset/mnist',
                               train=True,
                               transform=transform,
                               download=True)
test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              transform=transform,
                              download=True)
trains_loader = DataLoader(dataset=train_dataset,
                           batch_size=batch_size,
                           shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=32,
                         shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)    # 类似reshape函数
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        return self.l5(x)

model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_index, data in enumerate(trains_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index % 300 == 299:
            print(epoch+1, batch_index+1, running_loss/300)
            running_loss = 0.0

def test():
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)                       # dim = 0，指定的是行，那就是列不变
            _, predicted = torch.max(outputs.data, dim=1)    # dim=1，指定列，也就是行不变，列之间的比较
            total += labels.size(0)
            print(labels.size(0))
            correct += (predicted == labels).sum().item()
        print('correct rate: ', correct/total)


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()