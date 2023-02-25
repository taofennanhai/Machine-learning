import torch
import torch.nn as nn
import pandas as pd


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.a = nn.Parameter(torch.Tensor(1, 1))
        self.b = nn.Parameter(torch.Tensor(1, 1))
        self.c = nn.Parameter(torch.Tensor(1, 1))

    def forward(self, w, S):
        pred = self.a * w * (1 - w * self.b) + self.c * S * w
        return pred


df = pd.read_excel('Q3Data/test1.xlsx')

data_dwdt = df['dwdt'].values
data_w = df['w'].values
data_S = df['S'].values


# def Normalize(data):
#     mx = max(data)
#     mn = min(data)
#     return [(float(i) - mn) / (mx - mn) for i in data], mx, mn
#
# data_dwdt = Normalize()
# data_w =

data_dwdt = torch.tensor(data_dwdt).to(torch.float32)
data_w = torch.tensor(data_w).to(torch.float32)
data_S = torch.tensor(data_S).to(torch.float32)


model = Model()

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)


for epoch in range(100):
    for i in range(data_dwdt.shape[0]):

        predict = model(data_w[i], data_S[i])

        loss = torch.sqrt(loss_function(predict, data_dwdt[i]))

        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        print('RMSE损失为：', loss.item())

torch.save(model.state_dict(), 'result1.pkl')

for parameters in model.parameters():
    print(parameters)
