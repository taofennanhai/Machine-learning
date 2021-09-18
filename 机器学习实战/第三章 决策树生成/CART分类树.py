import numpy as np
import torch
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import ID3算法 as ID3
import pydotplus
import pandas as pd
import os
os.environ["PATH"] += os.pathsep + 'D:/Python/Lib/Graphviz/bin/'

dataSet, label = ID3.createDataSet()

x_data = []
y_data = []
for data in dataSet:
    x = data[:-1]
    y = data[-1]
    x_data.append(x)
    y_data.append(y)

x_data = pd.DataFrame(x_data)
y_data = pd.DataFrame(y_data)

x_data.rename(columns={0: '年龄', 1: '有工作', 2: '有自己的房子', 3: '信贷情况'}, inplace=True)
y_data.rename(columns={0: '是否借贷'}, inplace=True)
print(y_data)


clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(x_data.values, y_data['是否借贷'])
test = y_data['是否借贷']


print(clf.predict([[0, 1, 0, 1]]))

labels = ['age', 'job', 'home', 'Credit status']
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=labels, class_names='loan',
                                    filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('judge.pdf')






