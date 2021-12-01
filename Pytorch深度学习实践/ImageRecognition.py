import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


batch_size = 64    # 需要H*W*C的图转变为C*W*H的图
transform = transforms.Compose([transforms.ToTensor(),              # 把图像变为W*H，取值为0-1
                                transforms.Normalize((0.1307, ),    # 第一个值为均值mean，第二个为标准差std
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


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5), stride=1, bias=False)    # 第一个1为一个通道,之后变为10个通道,使用5*5的卷积核
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=(5, 5))    # 第二个卷积层
        self.pooling = torch.nn.MaxPool2d(kernel_size=(2, 2))    # 池化层使用2*2的格子
        self.fc = torch.nn.Linear(320, 10)    # 全连接层

    def forward(self, x):
        # 把x从数据（n, 1,28,28）转为 (n, 320)的数据形状
        batch_size = x.shape[0]
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = x.view(batch_size, -1)    # 拉成（batch_size,320）的形状
        x = self.fc(x)

        return x


model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)    # 把模型参数和缓存放入cuda中,迁移GPU中


def train(epoch):
    running_loss = 0.0
    for batch_index, data in enumerate(trains_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
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
            images, labels = data                         # 一个batch中又10000个样例
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)                       # dim = 0，指定的是行，那就是列不变
            _, predicted = torch.max(outputs.data, dim=1)    # dim=1，指定列，也就是行不变，列之间的比较
            total += labels.size(0)
            print(labels.size(0))
            correct += (predicted == labels).sum().item()
        print('correct rate: ', correct/total)


if __name__ == '__main__':
    print(torch.cuda.is_available())  # 判断是否可以使用gpu计算
    print(torch.cuda.device_count())  # 显示gpu数量
    print(torch.cuda.current_device())  # 当前使用gpu的设备号

    for epoch in range(10):
        train(epoch)
        test()