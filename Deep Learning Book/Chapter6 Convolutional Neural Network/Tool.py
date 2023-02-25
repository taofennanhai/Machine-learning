import torch
import torchvision
import pylab
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from d2l import torch as d2l
import time


class Accumulator:
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


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """画一系列图片"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i, (img, label) in enumerate(zip(imgs, titles)):
        xloc, yloc = i//num_cols, i % num_cols
        if torch.is_tensor(img):
            # 图片张量
            axes[xloc, yloc].imshow(img.reshape((28, 28)).numpy())
        else:
            # PIL图片
            axes[xloc, yloc].imshow(img)
        # 设置标题并取消横纵坐标上的刻度
        axes[xloc, yloc].set_title(label)
        plt.xticks([], ())
        axes[xloc, yloc].set_axis_off()
    pylab.show()


class Animator:  #@save
    """绘制数据"""
    def __init__(self, legend=None):
        self.legend = legend
        self.X = [[], [], []]
        self.Y = [[], [], []]

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

    def show(self):
        plt.plot(self.X[0], self.Y[0], 'r--')
        plt.plot(self.X[1], self.Y[1], 'g--')
        plt.plot(self.X[2], self.Y[2], 'b--')
        plt.legend(self.legend)
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.title('Visual')
        plt.show()
