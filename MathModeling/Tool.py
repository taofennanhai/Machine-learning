import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def Normalize(data):
    mx = max(data)
    mn = min(data)

    return [(float(i) - mn) / (mx - mn) for i in data], mx, mn


def getparameter():
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

    target = pd.read_excel('SecondQuestionData/annex3.xls', index_col=0)  # 下面是目标的归一化

    # test = np.array(temp.iloc[7:])
    data = np.array(target)[3:, 3:]

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

    return max_min, tar_max_min



def read_target():

    target = pd.read_excel('SecondQuestionData/annex3.xls', index_col=0)  # 下面是目标的归一化

    # test = np.array(temp.iloc[7:])
    data = np.array(target)[3:, 3:]

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

    return train_target, tar_max_min
