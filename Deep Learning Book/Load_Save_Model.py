import torch
from torch import nn
from torch.nn import functional as F

#
# x = torch.arange(4)
# torch.save(x, 'x-file')    # 一维张量的保存
#
# x2 = torch.load('x-file')    # 张量的读取
# print(x2)


class MLP(nn.Module):    # 演示模型参数的保存
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), 'mlp.params')    # 模型的参数

clone = MLP()    # 新的模型加载上一次的参数
clone.load_state_dict(torch.load('mlp.params'))    # 加载参数
clone.eval()

Y_clone = clone(X)
print(Y_clone == Y)