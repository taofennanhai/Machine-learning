import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):

    if not torch.is_grad_enabled():    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)    # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差

    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:   # 使用全连接层的情况，计算特征维上的均值和方差
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)

            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:    # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

        X_hat = (X - mean) / torch.sqrt(var + eps)    # 训练模式下，用当前的均值和方差做标准化

        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean    # 更新移动平均的均值和方差
        moving_var = momentum * moving_var + (1.0 - momentum) * var

    Y = gamma * X_hat + beta  # 缩放和移位

    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):

    def __init__(self, num_features, num_dims):    # num_features：全连接层的输出数量或卷积层的输出通道数。 num_dims：2表示完全连接层，4表示卷积层
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        self.gamma = nn.Parameter(torch.ones(shape))    # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.beta = nn.Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)    # 非模型参数的变量初始化为0和1
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1)),    # 6*(28-5+2*2+1)/1 = 6*28
            BatchNorm(6, num_dims=4), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),    # 6*(28-2+2)/2 = 6*14

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size= (5, 5), padding=(0, 0), stride=(1, 1)),    # 16*(14-5+1)/1 = 16*10
            BatchNorm(16, num_dims=4), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),    # 16*(10-2+2)/2 = 6*5

            nn.Flatten(),    # torch.nn.Flatten()默认从第二维开始平坦化。
            nn.Linear(16 * 5 * 5, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
            nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
            nn.Linear(84, 10)
        )

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)

        self.net.apply(init_weights)

    def forward(self, X):

        # for layer in self.net:
        #     X = layer(X)
        #     print(layer.__class__.__name__, 'output shape: \t', X.shape)
        X = self.net(X)
        return X


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
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


model = LeNet()

# X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# model(X)

lr = 0.9
batch_size = 256
device = 'cpu'
num_epochs = 10

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=0)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=0)


model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

loss = nn.CrossEntropyLoss()

animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])    # 画图
# animator = Animator(legend=['train loss', 'train acc', 'test acc'])

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
        animator.add(epoch + 1, (None, None, test_acc))
plt.show()


print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')