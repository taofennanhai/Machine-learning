import os

import graphviz
import numpy as np
import pydotplus
from sklearn.tree import DecisionTreeRegressor, export_graphviz
os.environ["PATH"] += os.pathsep + 'D:/Anaconda3/envs/Python/Lib/Graphviz/bin'

def loadDataSet(fileName):
    """导入数据"""
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 使用python3会报错1，因为python3中map的返回类型是‘map’类，不能进行计算，需要将map转换为list
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


if __name__ == '__main__':
    dataMat = np.array(loadDataSet('ex0.txt'))

    X = dataMat[:, 0:2]
    y = dataMat[:, -1]

    clf = DecisionTreeRegressor().fit(X, y)
    dot_data = export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('judge.pdf')
