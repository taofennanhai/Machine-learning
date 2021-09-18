import torch
import math
import torchvision

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0.0], [0.0], [1.0]])

# class LogisticRegressionModel(torch.nn.Module):
#     def __init__(self):
#         super(LogisticRegressionModel, self).__init__()    #调用父类的构造函数，
#         self.linear = torch.nn.Linear(1, 1)    #in_features：前一层网络神经元的个数 out_features： 该网络层神经元的个数以上两者决定了weight的形状[out_features , in_features]
#
#     def forward(self, x):
#         y_pred = torch.sigmoid(self.linear(x))
#         return y_pred
#
# model = LogisticRegressionModel()
# criterion = torch.nn.BCELoss(reduction='mean')    #需要y_hat和y
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# for epoch in range(1000):    #第一步先算y_hat，第二部分算损失，第三步一就是反向传播，最后一步是梯度更新
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     print(epoch, loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# print(' w=', model.linear.weight.item())
# print(' b=', model.linear.bias.item())
#
# x_test = torch.Tensor([[4.0]])
# y_test = model.forward(x_test)
# print('y_test:', y_test.item())



pred = torch.tensor([[-0.2], [0.2], [0.8]])
target = torch.tensor([[0.0], [0.0], [1.0]])

sigmoid = torch.nn.Sigmoid()
pred_s = sigmoid(pred)
print(pred_s)
"""
pred_s 输出tensor([[0.4502],[0.5498],[0.6900]])
0*math.log(0.4502)+1*math.log(1-0.4502)
0*math.log(0.5498)+1*math.log(1-0.5498)
1*math.log(0.6900) + 0*log(1-0.6900)
"""
result = 0
i = 0
for label in target:
    if label.item() == 0:
        result += math.log(1 - pred_s[i].item())
    else:
        result += math.log(pred_s[i].item())
    i += 1
result /= 3
print("bce：", -result)
loss = torch.nn.BCELoss()
print('BCELoss:', loss(pred_s, target).item())
