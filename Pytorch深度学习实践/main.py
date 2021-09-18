# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')


import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# def forward(x, b):    #模型是y=x*w+b,也就是前向传播的过程
#     return x*w+b
#
# def loss(x, y, b):    #损失函数（y预测-y真实）^2
#     y_pred = forward(x, b)
#     return (y_pred-y)*(y_pred-y)
#
# w_list = []
# b_list = []
# mse_list = []
#
# for w in np.arange(0.0, 4, 0.1):
#     for b in np.arange(-2, 2, 0.1):
#         print('w=', w, ' b=', b)
#         l_sum = 0
#         for x_val, y_val in zip(x_data, y_data):
#             y_pred_val = forward(x_val, b)
#             loss_val = loss(x_val, y_val, b)
#             l_sum += loss_val
#             print('\t', x_val, y_val, b, y_pred_val, loss_val)
#         print('MSE=', l_sum / 3)
#         w_list.append(w)
#         b_list.append(b)
#         mse_list.append(l_sum / 3)


# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# ax1.scatter3D(w_list, b_list, mse_list, cmap= 'Blues') #绘制散点图
# ax1.plot3D(w_list, b_list, mse_list, 'gray') #绘制空间曲线
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = np.array(w_list).reshape((40, 40))
# y = np.array(b_list).reshape((40, 40))
# z = np.array(mse_list).reshape((40, 40))
#
# ax.plot_surface(x, y, z)
# ax.set_xlabel('\n' + 'Weight', linespacing=4)
# ax.set_ylabel('B')
#
# plt.show()


x = torch.Tensor(2, 3)
print(x)
y = torch.Tensor(2, 3)
print(y)

print(torch.cuda.is_available())

x = x.cuda()
y = y.cuda()
print(x+y)