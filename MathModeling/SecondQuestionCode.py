import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
from Tool import getparameter
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

matplotlib.rc('axes', facecolor='white')
matplotlib.rc('figure', figsize = (6, 4))
matplotlib.rc('axes', grid = False)


def Normalize(data):
    mx = max(data)
    mn = min(data)

    return [(float(i) - mn) / (mx - mn) for i in data], mx, mn


all_data = pd.read_excel('SecondQuestionData/annex8/' + str(2012) + '.xls')

# test = np.array(temp.iloc[7:])
data1, data2, = np.array(all_data)[:, 6:12], np.array(all_data)[:, 13:21]
data3, data4 = np.array(all_data)[:, 22:28], np.array(all_data)[:, 30:]

data = np.concatenate((data1, data2, data3, data4), axis=1)


for year in range(2013, 2022):

    all_data = pd.read_excel('SecondQuestionData/annex8/' + str(year) + '.xls')

    # test = np.array(temp.iloc[7:])
    data1, data2, = np.array(all_data)[:, 6:12], np.array(all_data)[:, 13:21]
    data3, data4 = np.array(all_data)[:, 22:28], np.array(all_data)[:, 30:]

    temp = np.concatenate((data1, data2, data3, data4), axis=1)

    data = np.concatenate((data, temp), axis=0)

data = data.T
max_min = []
for i in range(0, 22):
    temp = []
    data[i], max_num, min_num = Normalize(data[i])
    temp.append(max_num)
    temp.append(min_num)
    max_min.append(temp)

    # train_data.append(pd.read_excel('SecondQuestionData/annex8/' + str(year) + '.xls', index_col=0))
train_data = data.T


target = pd.read_excel('SecondQuestionData/annex3.xls', index_col=0)    # 下面是目标的归一化

# test = np.array(temp.iloc[7:])
data = np.array(target)[3:, 3:][::-1]

data = data.T
tar_max_min = []

for i in range(0, 4):
    temp = []
    data[i], max_num, min_num = Normalize(data[i])
    temp.append(max_num)
    temp.append(min_num)
    tar_max_min.append(temp)

    # train_data.append(pd.read_excel('SecondQuestionData/annex8/' + str(year) + '.xls', index_col=0))
train_target = data.T


lstmcell = nn.GRUCell(22, 4, bias=True)

optimizer = torch.optim.Adam(lstmcell.parameters(), lr=0.0005)

loss_function = nn.MSELoss()

train_data = torch.tensor(train_data).to(torch.float32)
train_target = torch.tensor(np.ascontiguousarray(train_target)).to(torch.float32)
# train_target = torch.tensor(train_target)


past_target = torch.zeros(1, 4)

for i in range(10):
    index = i * 12 + 11
    test = train_target[index]
    past_target += train_target[index]

past_target /= 10


x = []
for i in range(2012, 2022):
    for j in range(1, 13):
        x.append(str(i) + ' year ' + str(j) + ' mouth')

loss_list = []

for epoch in range(100):

    predict_10_labels = []
    predict_40_labels = []
    predict_100_labels = []
    predict_200_labels = []


    for i in range(120):

        pred_target = lstmcell(train_data[i].unsqueeze(0), past_target)
        predict_10_labels.append(pred_target.squeeze(0)[0].tolist())
        predict_40_labels.append(pred_target.squeeze(0)[1].tolist())
        predict_100_labels.append(pred_target.squeeze(0)[2].tolist())
        predict_200_labels.append(pred_target.squeeze(0)[3].tolist())

        loss = torch.sqrt(loss_function(train_target[i], pred_target.squeeze(0)))
        loss_list.append(loss.item())


        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        past_target = train_target[i].unsqueeze(0)

        print('RMSE损失为：', loss.item())

    # test = predict_10_labels
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False  # 负号显示
    # # 数据及线属性
    # plt.plot(x, predict_100_labels, color='blue', marker='o', linestyle='-', label='pred')
    # plt.plot(x, train_target[:, 2], color='red', marker='D', linestyle='-.', label='true')
    #
    # plt.legend()  # 显示图例
    # plt.xlabel("data")  # X轴标签
    # plt.ylabel("predict")  # Y轴标签
    # plt.show()


y_train_loss = loss_list
x_train_loss = range(len(y_train_loss))

plt.figure()

# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('iters')  # x轴标签
plt.ylabel('loss')  # y轴标签

# 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
plt.legend()
plt.title('Loss curve')
plt.show()


# torch.save(lstmcell.state_dict(), 'target_result3.pkl')

print('yes')