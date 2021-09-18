import torch
import numpy as np
import matplotlib.pyplot as plt

w=1
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):    #模型是y=x*w+b,也就是前向传播的过程  梯度下降，拿整个来计算
    return x*w

def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred-y)**2
    return cost/len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2*x*(x*w-y)
    return grad/len(xs)

def loss(x, y):    #随机梯度下降定义损失
    y_pred = forward(x)
    return (y_pred-y)**2

def stochastic_gradient(x,y):    #拿一个点的梯度计算
    return 2*x*(x*w-y)



# print('Predict(before Training)', 4, forward(4))    #梯度下降
# for epoch in range(100):
#     cost_val = cost(x_data, y_data)
#     grad_val = gradient(x_data, y_data)
#     w -= 0.01*grad_val
#     print('Epoch:', epoch, ' w=', w, ' Loss=', cost_val)
# print('Predict(after Training)', 4, forward(4))

epoch_list = []
loss_list = []
print('Predict(before Training)', 4, forward(4))    #随机梯度下降
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = stochastic_gradient(x, y)
        w = w-0.01*grad
        print('\tgrad:', x, y, grad)
        l = loss(x, y)

    print('process:', epoch, ' w=', w, ' loss:', l)
    epoch_list.append(epoch)
    loss_list.append(l)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()