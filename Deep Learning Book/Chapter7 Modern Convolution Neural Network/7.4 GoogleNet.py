from torch import nn
from d2l import torch as d2l
import torch
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


class Inception(nn.Module):

    def __init__(self, in_channels, c1, c2, c3, c4):    # c1--c4是每条路径的输出通道数
        super(Inception, self).__init__()

        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)    #线路1，单1x1卷积层

        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)    # 线路2，1x1卷积层后接3x3卷积层
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)    # 线路3，1x1卷积层后接5x5卷积层
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)    #线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)  # 线路4，3x3最大汇聚层后接1x1卷积层

    def forward(self, x):

        p1 = nn.functional.relu(self.p1_1(x))    # 第一条路径的输入

        p2 = nn.functional.relu(self.p2_2(nn.functional.relu(self.p2_1(x))))  # 第二条路径的输入

        p3 = nn.functional.relu(self.p3_2(nn.functional.relu(self.p3_1(x))))  # 第三条路径的输入

        p4 = nn.functional.relu(self.p4_2(self.p4_1(x)))  # 第四条路径的输入

        return torch.cat((p1, p2, p3, p4), dim=1)    # 在通道维度上连结输出


class GoogLeNet(nn.Module):

    def __init__(self):
        super(GoogLeNet, self).__init__()

        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))    # 第一个模块使用64个通道、7*7卷积层

        b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                           nn.ReLU(),
                           nn.Conv2d(64, 192, kernel_size=3, padding=1),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))    # 第二个模块使用两个卷积层：第一个卷积层是64个通道、1*1卷积层；第二个卷积层使用将通道数量增加三倍的3*3卷积层

        b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                           Inception(256, 128, (128, 192), (32, 96), 64),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))    # 第三个模块串联两个完整的Inception块   第二个和第三个路径首先将输入通道的数量分别减少到96/192 和 16/192

        b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                           Inception(512, 160, (112, 224), (24, 64), 64),
                           Inception(512, 128, (128, 256), (24, 64), 64),
                           Inception(512, 112, (144, 288), (32, 64), 64),
                           Inception(528, 256, (160, 320), (32, 128), 128),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))    # 第四模块串联了5个Inception块

        b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                           Inception(832, 384, (192, 384), (48, 128), 128),
                           nn.AdaptiveAvgPool2d((1, 1)),
                           nn.Flatten())    # 第五模块包含输出通道数为832和1024, 该模块同NiN一样使用全局平均汇聚层，将每个通道的高和宽变成1

        self.net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))   # 输出变成二维数组，再接上一个输出个数为标签类别数的全连接层
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, X):
        for layer in self.net:
            X = layer(X)
        return X


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

model = GoogLeNet()
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