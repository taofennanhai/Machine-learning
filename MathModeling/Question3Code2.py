# 本代码用的回归模型是随机森林
import xlrd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pylab import mpl

#mpl.rcParams['font.sans-serif'] = ['SimHei']

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']    # 指定默认字体：解决plot不能显示中文问题

mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题
warnings.filterwarnings('ignore')

df = pd.read_excel('Q3Data/q3datas.xlsx')

df_coor = df.corr()
print(df_coor)


fig, ax = plt.subplots(figsize=(6, 6), facecolor='w')
# 指定颜色带的色系
sns.heatmap(df.corr(), annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g')
plt.title('相关性热力图')
plt.show()










