import torch
from torch.nn import functional as F
from torch import nn
from d2l import torch as d2l
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt


class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Residual(nn.Module):  #@save

    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):

        Y1 = F.relu(self.bn1(self.conv1(X)))

        Y = self.bn2(self.conv2(Y1))

        if self.conv3:
            X = self.conv3(X)

        Y += X

        return F.relu(Y)


# blk = Residual(3, 3)
# X = torch.rand(4, 3, 6, 6)
# Y = blk(X)
# print(Y.shape)


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 第一个模块使用64个通道、7*7卷积层

        b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))

        b3 = nn.Sequential(*self.resnet_block(64, 128, 2))

        b4 = nn.Sequential(*self.resnet_block(128, 256, 2))

        b5 = nn.Sequential(*self.resnet_block(256, 512, 2))

        self.net = nn.Sequential(b1, b2, b3, b4, b5,
                                 nn.AdaptiveAvgPool2d((1, 1)),    # 把特征图变成1*1的矩阵
                                 nn.Flatten(),
                                 nn.Linear(512, 10))

    def resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:    # 不是第一个块的话就使用1*1卷积残差
                blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))    # 每个模块在第一个残差块里将高和宽减半
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    def forward(self, X):

        for layer in self.net:
            X = layer(X)
            # print(layer.__class__.__name__, 'output shape:\t', X.shape)

        return X


# X = torch.rand(size=(1, 1, 224, 224))
#
# X = resnet(X)
#

lr, num_epochs, batch_size = 0.05, 10, 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize = 96

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0到1之间
trans = [transforms.ToTensor()]
trans.insert(0, transforms.Resize(resize))
trans = transforms.Compose(trans)
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=0)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=0)

model = ResNet()
model = model.to(device)

# X = torch.rand(size=(1, 1, 96, 96)).to(device)
#
# X1 = model(X)
#
# print(X1)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

loss = nn.CrossEntropyLoss()

animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])    # 画图

num_batches = len(train_iter)

for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        model.train()
        for i, (X, y) in enumerate(train_iter):

            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])

            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
                # animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy(model, test_iter)
        print(test_acc)
        animator.add(epoch + 1, (None, None, test_acc))
plt.show()


print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')

