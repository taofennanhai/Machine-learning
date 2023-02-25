import xlrd
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')


data = pd.read_excel('Q3Data/q3datas.xlsx', 'Sheet1')

data_y1 = data['SIC土壤无机碳'].values.reshape(-1, 1)
data_y2 = data['土壤C/N比']
data_x = data['生物量干重']

y1 = sm.add_constant(data_y1)

model = sm.OLS(data_x, y1)

result = model.fit()

print(result.summary())
print(result.params)
# data_y=pd.Series(random.sample(range(200),10))

