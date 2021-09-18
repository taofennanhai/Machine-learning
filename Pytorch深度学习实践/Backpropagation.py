import torch
import numpy as np

x_data = [1, 2, 3]
y_data = [2, 4, 6]

# w = torch.Tensor([1.0])
# w.requires_grad = True
# def forward(x):    #假设模型是x*w
#     return x*w
#
# def loss(x, y):
#     y_pred = forward(x)
#     return (y_pred - y) ** 2
#
#
# print('Predict(before Training)', 4, forward(4).item())
#
# for epoch in range(100):
#     for x, y in zip(x_data, y_data):
#         l = loss(x, y)
#         l.backward()
#         print('\tgrad:', x, y, w.grad.item())
#         w.data = w.data - 0.01*w.grad.data
#
#         w.grad.data.zero_()
#     print("progress:", epoch, l.item())
#
# print('predict (after training)', 4, forward(4).item())

w1 = torch.Tensor([1.0])
w2 = torch.Tensor([2.0])
b = torch.tensor([3.0])

w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True

# print('w1:', w1, 'w2', w2)    #随机梯度下降
# def forward(x, b):
#     return w1*x*x+w2*x+b
#
# def loss(x, b, y):
#     y_pred = forward(x, b)
#     return (y_pred - y)**2
#
# print('Predict(before Training)', 4, forward(4, 1).item())
#
# for epoch in range(100):
#     for x, y in zip(x_data, y_data):
#         l = loss(x, b, y)
#         l.backward()
#         print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
#         w1.data = w1.data - 0.01*w1.grad.data
#         w2.data = w2.data - 0.01 * w2.grad.data
#         b.data = b.data - 0.01 * b.grad.data
#
#         w1.grad.data.zero_()
#         w2.grad.data.zero_()
#         b.grad.data.zero_()
#     print("progress:", epoch, l.item())
#
# print('predict (after training)', 4, forward(4, 0.16).item())

def forward(x, b):    #定义模型为w1*x*x+w2*x+b  使用批梯度下降
    return w1*x*x+w2*x+b

def cost(xs, ys):
    cost = torch.Tensor([0.0])
    for x, y in zip(xs, ys):
        y_pred = forward(x, b)
        cost += (y_pred-y)**2
    return cost/len(xs)

print('Predict(before Training)', 4, forward(4, 1).item())
for epoch in range(1000):
    cost_val = cost(x_data, y_data)
    cost_val.backward()
    w1.data = w1.data - 0.01 * w1.grad.data
    w2.data = w2.data - 0.01 * w2.grad.data
    b.data = b.data - 0.01 * b.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()
    b.grad.data.zero_()

    print('Epoch:', epoch, ' w1=', w1.item(), ' w2=', w2.item(), ' b=', b.item(), ' Loss=', cost_val.item())
print('Predict(after Training)', 4, forward(4, 0).item())
