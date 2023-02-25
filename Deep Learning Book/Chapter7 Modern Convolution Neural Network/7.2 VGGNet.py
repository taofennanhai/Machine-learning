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


class VGGNet(nn.Module):

    def __init__(self, conv_arch):
        super(VGGNet, self).__init__()

        if conv_arch == None or conv_arch == []:
            conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))    # 第一个维度是卷积层数量，第二个是通道数量
        conv_blks = []
        in_channels = 1

        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        self.net = nn.Sequential(
                *conv_blks, nn.Flatten(),
                # 全连接层部分
                nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),     # 7 * 7是特征图大小
                nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(4096, 10))
        print(self.net)

    def vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, X):
        for blk in self.net:
            X = blk(X)
            # print(blk.__class__.__name__, 'output shape:\t', X.shape)
        return X


X = torch.randn(size=(1, 1, 224, 224))

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))    # 第一个定义的是几个卷积层，第二个维度定义输出通道数量

VGG = VGGNet(conv_arch=conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize = 224


# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0到1之间
trans = [transforms.ToTensor()]
trans.insert(0, transforms.Resize(resize))
trans = transforms.Compose(trans)
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=0)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=0)

model = VGG.to(device)

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




