import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from Tool import getparameter, read_target


matplotlib.rc('axes', facecolor='white')
matplotlib.rc('figure', figsize = (6, 4))
matplotlib.rc('axes', grid = False)


def Normalize(data, mx=None, mn=None):

    if mx == None and mn == None:
        mx = max(data)
        mn = min(data)
    return [(float(i) - mn) / (mx - mn) for i in data], mx, mn


df = pd.read_excel('SecondQuestionData/annex4.xls', index_col=0)

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

train_data = data.T

feature_max_min = []
for i in range(0, 22):
    temp = []
    train_data[i], max_num, min_num = Normalize(train_data[i])
    temp.append(max_num)
    temp.append(min_num)
    feature_max_min.append(temp)

train_data = train_data.T


append_data = pd.read_excel('SecondQuestionData/annex8/2022.xls')    # 下面是目标的归一化
data1, data2, = np.array(append_data)[0, 6:12], np.array(append_data)[0, 13:21]
data3, data4 = np.array(append_data)[0, 22:28], np.array(append_data)[0, 30:]

tar_data = np.concatenate((data1, data2, data3, data4), axis=0)
tar_data = tar_data[None]

for i in range(0, 22):
    test1 = feature_max_min[i][0]
    test2 = feature_max_min[i][1]
    tar_data[0][i] = (tar_data[0][i] - feature_max_min[i][1]) / (feature_max_min[i][0] - feature_max_min[i][1])

tar_data = np.concatenate((data, tar_data))[1:, ]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstmcell = nn.GRUCell(22, 50, bias=True)
        self.linear = nn.Linear(50, 22, bias=True)

    def forward(self, x, past_info):

        info = self.lstmcell(x, past_info)
        pred = self.linear(info)

        cur_info = info

        return pred, cur_info

lstmcell = Model()

optimizer = torch.optim.Adam(lstmcell.parameters(), lr=0.0005)

loss_function = nn.MSELoss()

train_data = torch.tensor(train_data).to(torch.float32)
tar_data = torch.tensor(tar_data).to(torch.float32)


x = []
for i in range(2012, 2022):
    for j in range(1, 13):
        x.append(str(i) + ' year ' + str(j) + ' mouth')

info = torch.randn(1, 50)


for epoch in range(50):

    feature_labels = []
    predict_40_labels = []
    predict_100_labels = []
    predict_200_labels = []
    predict_target = torch.zeros(1, 22)

    for i in range(120):

        pred_target, cur_info = lstmcell(train_data[i].unsqueeze(0), info)

        loss = torch.sqrt(loss_function(tar_data[i], pred_target))

        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        print('RMSE损失为：', loss.item())
        predict_target = torch.cat((predict_target, pred_target), dim=0)

    predict_target = predict_target[1:, ]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 负号显示
    # 数据及线属性
    plt.plot(x, predict_target[:, 7].detach().numpy(), color='blue', marker='o', linestyle='-', label='pred')
    plt.plot(x, train_data[:, 7].detach().numpy(), color='red', marker='D', linestyle='-.', label='true')

    plt.legend()  # 显示图例
    plt.xlabel("date")  # X轴标签
    plt.ylabel("predict")  # Y轴标签
    plt.show()


max_min, tar_max_min = getparameter()

feature_model = nn.GRUCell(22, 4, bias=True)
feature_model.load_state_dict(torch.load('target_result3.pkl'))

train_target, tar_max_min = read_target()

pred_feature = train_data[119, ].unsqueeze(0)
pred_result = torch.tensor(train_target[119, ]).unsqueeze(0).to(torch.float32)

norm_scaler = MinMaxScaler()
norm_data = norm_scaler.fit_transform(data)

for i in range(24):

    pred_target, cur_info = lstmcell(pred_feature, info)
    pred_result = feature_model(pred_target, pred_result)

    pred_feature = pred_target
    info = cur_info


    result10 = pred_result.detach().numpy()[0][0]
    result40 = pred_result.detach().numpy()[0][1]
    result100 = pred_result.detach().numpy()[0][2]
    result200 = pred_result.detach().numpy()[0][3]

    x10 = result10 * (tar_max_min[0][0]-tar_max_min[0][1]) + tar_max_min[0][1]
    x40 = result40 * (tar_max_min[1][0] - tar_max_min[1][1]) + tar_max_min[1][1]
    x100 = result100 * (tar_max_min[2][0] - tar_max_min[2][1]) + tar_max_min[2][1]
    x200 = result200 * (tar_max_min[3][0] - tar_max_min[3][1]) + tar_max_min[3][1]

    print('predict10 is ', x10, 'predict40 is ', x40, 'predict100 is ', x100, 'predict200 is ', x200)



