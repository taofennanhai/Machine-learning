# 本代码用的回归模型是随机森林
import xlrd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


f_name = 'Q3Data/annex14.xlsx'

soild_data = np.array(pd.read_excel('Q3Data/annex14.xlsx'))


epoch = 0.0
average_value1 = 0.0
average_value2 = 0.0
average_value3 = 0.0
average_value4 = 0.0
average_value5 = 0.0
for i in range(soild_data.shape[0]):

    if soild_data[i][0] == 2018 and soild_data[i][1] == 'G18':
        average_value1 += soild_data[i][3]
        average_value2 += soild_data[i][4]
        average_value3 += soild_data[i][5]
        average_value4 += soild_data[i][6]
        average_value5 += soild_data[i][7]
        epoch += 1
        print(soild_data[i][3], ' ', soild_data[i][4], ' ', soild_data[i][5], ' ', soild_data[i][6])

print(average_value1/epoch)
print(average_value2/epoch)
print(average_value3/epoch)
print(average_value4/epoch)
print(average_value5/epoch)


print('----------------------')

soild_data = np.array(pd.read_excel('Q3Data/annex15.xlsx'))


epoch = 0.0
average_value1 = 0.0
average_value2 = 0.0
average_value3 = 0.0
average_value4 = 0.0
average_value5 = 0.0
for i in range(soild_data.shape[0]):

    if soild_data[i][0] == 2018 and soild_data[i][1] == 'G18':
        average_value1 += soild_data[i][3]
        average_value2 += soild_data[i][4]
        average_value3 += soild_data[i][5]
        average_value4 += soild_data[i][6]
        average_value5 += soild_data[i][7]
        epoch += 1
        print(soild_data[i][3], ' ', soild_data[i][4], ' ', soild_data[i][5], ' ', soild_data[i][6])

print(average_value1/epoch)
print(average_value2/epoch)
print(average_value3/epoch)
print(average_value4/epoch)















