from torch import nn
from d2l import torch as d2l
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(11, 11), padding=(1, 1), stride=(4, 4)),   # 96*(24-11+2+4)/4 = 96*54
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),    # 96*(54-3+2)/2 = 96*26
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=(2, 2), stride=(1, 1)),    # 256*(26-5+4+1)/1 = 256*26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),    # 256*(26-3+2)/2 = 256*12
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),    # 384*(12-3+2+1)/1 = 384*12
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),  # 384*(12-3+2+1)/1 = 384*12
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),  # 256*(12-3+2+1)/1 = 256*12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),    # 256*(12-3+2)/2 = 256*5
            nn.Flatten(),    # torch.nn.Flatten()默认从第二维开始平坦化。
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
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
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

d2l.train_ch6
model = AlexNet()

# X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# model(X)

lr = 0.01
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
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
# print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')