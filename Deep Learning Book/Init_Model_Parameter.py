import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):    # 演示模型参数的保存
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
        self.lstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=1, bidirectional=False)



    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


def weights_init(m):    # 权重初始化
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

    if isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

net.apply(weights_init)

for name, parameters in net.named_parameters():    # 查看网络中的参数
    print(name)
    print(parameters)
    print(parameters.shape)



