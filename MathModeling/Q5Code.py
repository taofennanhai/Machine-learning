import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import statsmodels.api as sm

df = pd.read_excel('Q5Data/DryWeight.xlsx')

df2 = pd.read_excel('Q5Data/ICstore.xlsx')


# print(df[df['blocks'] == 'G17'])
#
# print(df[df['blocks'] == 'G19'])
#
# print(df[df['blocks'] == 'G21'])
#
# icstore2019 = df2['ICstore'][16:21, ]
# icstore2020 = df2['ICstore'][4:9, ]
# print(icstore2019)
# print(icstore2020)
#
#
# y = np.array([])
#
# y = np.concatenate([y, df[df['blocks'] == 'G17']['生物量(干重)'].values], axis=0)
# y = np.concatenate([y, df[df['blocks'] == 'G19']['生物量(干重)'].values], axis=0)
# y = np.concatenate([y, df[df['blocks'] == 'G21']['生物量(干重)'].values], axis=0)/1000 * 14400
#
# x = np.array([])
#
# for i in range(3):
#     x = np.concatenate([x, icstore2019.values], axis=0)
#     x = np.concatenate([x, icstore2020.values], axis=0)
# temp = x
# import statsmodels.api as sm
#
# x = sm.add_constant(x)  # 线性回归增加常数项
# est = sm.OLS(y, x).fit()
#
# est = sm.OLS(y, x).fit()
# print(est.summary())
#
#
# test = 3284.25 * x[:, 0] + 684.71
#
# a_ic = 3284.25 * x[:, 0]
# c = 684.71
#
# # x = np.arange(-100, 100, 1) #定义x的范围为-100至100，步长为1
# plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# plt.xlabel('x') #绘制X轴
# plt.ylabel('y') #绘制Y轴
# plt.title("y = x * x") #绘制图像标题
# plt.plot(temp, y, 'o')
# plt.plot(x, 3284.25 * x[:, 0] + 684.71, "r--")
# plt.show()






# w = np.array([])
#
# light6 = df[df['blocks'] == 'G6']
# light12 = df[df['blocks'] == 'G12']
# light18 = df[df['blocks'] == 'G18']
#
# test = np.concatenate((light6['生物量(干重)'].values[1:5], light6['生物量(干重)'].values[6:]), axis=0)
# w = np.concatenate([w, np.concatenate((light6['生物量(干重)'].values[1:5], light6['生物量(干重)'].values[6:]), axis=0)], axis=0)
# w = np.concatenate([w, np.concatenate((light12['生物量(干重)'].values[1:5], light12['生物量(干重)'].values[6:]), axis=0)], axis=0)
# w = np.concatenate([w, np.concatenate((light18['生物量(干重)'].values[1:5], light18['生物量(干重)'].values[6:]), axis=0)], axis=0)*14.400
#
# icstore2019 = df2['ICstore'][17:21, ]
# icstore2020 = df2['ICstore'][5:9, ]
# print(icstore2019)
# print(icstore2020)
#
# x1 = []
# for i in range(3):
#     x1 = np.concatenate([x1, np.concatenate([icstore2019, icstore2020], axis=0)], axis=0)
#
# sc = np.array([])
# b = 2
# a = 1
# for i in range(24):
#     sheep = (b - a) * np.random.random_sample() + a
#     sc = np.concatenate([sc, np.array([1.44 * sheep* 3 * 1.8])], axis=0)
#
#
# # mid8 = df[df['blocks'] == 'G8']
# # mid11 = df[df['blocks'] == 'G11']
# # mid16 = df[df['blocks'] == 'G16']
# #
# # test = np.concatenate((mid8['生物量(干重)'].values[1:5], light6['生物量(干重)'].values[6:]), axis=0)
# # w = np.concatenate([w, np.concatenate((light6['生物量(干重)'].values[1:5], light6['生物量(干重)'].values[6:]), axis=0)], axis=0)
#
# x = np.c_[x1, sc]
# x_model = sm.add_constant(x)
# model = sm.OLS(w, x_model)
#
# results = model.fit()# fit拟合
# print(results.summary())# summary方法主要是为了显示拟合的结果



# w = np.array([])
#
# mid8 = df[df['blocks'] == 'G8']
# mid11 = df[df['blocks'] == 'G11']
# mid16 = df[df['blocks'] == 'G16']
#
# # test = np.concatenate((light6['生物量(干重)'].values[1:5], light6['生物量(干重)'].values[6:]), axis=0)
#
# w = np.concatenate([w, np.concatenate((mid8['生物量(干重)'].values[1:5], mid8['生物量(干重)'].values[6:]), axis=0)], axis=0)
# w = np.concatenate([w, np.concatenate((mid11['生物量(干重)'].values[1:5], mid11['生物量(干重)'].values[6:]), axis=0)], axis=0)
# w = np.concatenate([w, np.concatenate((mid16['生物量(干重)'].values[1:5], mid16['生物量(干重)'].values[6:]), axis=0)], axis=0)*14.400
#
# icstore2019 = df2['ICstore'][17:21, ]
# icstore2020 = df2['ICstore'][5:9, ]
# print(icstore2019)
# print(icstore2020)
#
# x2 = []
# for i in range(3):
#     x2 = np.concatenate([x2, np.concatenate([icstore2019, icstore2020], axis=0)], axis=0)
#
# sc = np.array([])
# b = 4
# a = 3
# for i in range(24):
#     sheep = (b - a) * np.random.random_sample() + a
#     sc = np.concatenate([sc, np.array([1.44 * sheep * 6 * 1.8])], axis=0)
#
# x = np.c_[x2, sc]
# x_model = sm.add_constant(x)
# model = sm.OLS(w, x_model)
#
# results = model.fit()# fit拟合
# print(results.summary())# summary方法主要是为了显示拟合的结果


w = np.array([])

high9 = df[df['blocks'] == 'G9']
high13 = df[df['blocks'] == 'G13']
high20 = df[df['blocks'] == 'G20']

# test = np.concatenate((light6['生物量(干重)'].values[1:5], light6['生物量(干重)'].values[6:]), axis=0)

w = np.concatenate([w, np.concatenate((high9['生物量(干重)'].values[1:5], high9['生物量(干重)'].values[6:]), axis=0)], axis=0)
w = np.concatenate([w, np.concatenate((high13['生物量(干重)'].values[1:5], high13['生物量(干重)'].values[6:]), axis=0)], axis=0)
w = np.concatenate([w, np.concatenate((high20['生物量(干重)'].values[1:5], high20['生物量(干重)'].values[6:]), axis=0)], axis=0)*14.400

icstore2019 = df2['ICstore'][17:21, ]
icstore2020 = df2['ICstore'][5:9, ]
print(icstore2019)
print(icstore2020)

x2 = []
for i in range(3):
    x2 = np.concatenate([x2, np.concatenate([icstore2019, icstore2020], axis=0)], axis=0)

sc = np.array([])
b = 8
a = 5
for i in range(24):
    sheep = (b - a) * np.random.random_sample() + a
    sc = np.concatenate([sc, np.array([1.44 * sheep * 12 * 1.8])], axis=0)

x = np.c_[x2, sc]
x_model = sm.add_constant(x)
model = sm.OLS(w, x_model)

results = model.fit()# fit拟合
print(results.summary())# summary方法主要是为了显示拟合的结果