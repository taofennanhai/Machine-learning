import torch
import numpy as np

x_data = torch.Tensor([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]])
y_data = torch.Tensor([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()    #调用父类的构造函数，
        self.linear = torch.nn.Linear(1, 1)    #in_features：前一层网络神经元的个数 out_features： 该网络层神经元的个数以上两者决定了weight的形状[out_features , in_features]

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()
criterion = torch.nn.MSELoss(reduction='mean')    #需要y_hat和y
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):    #第一步先算y_hat，第二部分算损失，第三步一就是反向传播，最后一步是梯度更新
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(' w=', model.linear.weight.item())
    print(' b=', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred:', y_test.item())

# 一般在反向传播时，都是先求loss，再使用loss.backward()求loss对每个参数 w_ij和b的偏导数(也可以理解为梯度)。
#
# 这里要注意的是，只有标量才能执行backward()函数，因此在反向传播中reduction不能设为'none'。
#
# 但具体设置为'sum'还是'mean'都是可以的。
#
# 若设置为'sum'，则有Loss=loss_1+loss_2+loss_3，表示总的Loss由每个实例的loss_i构成，在通过Loss求梯度时，将每个loss_i的梯度也都考虑进去了。
#
# 若设置为'mean'，则相比'sum'相当于Loss变成了Loss*(1/i)，这在参数更新时影响不大，因为有学习率a的存在。