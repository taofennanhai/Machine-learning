import math

import numpy as np
import pandas as pd


def loadDataSet(filePath):

    df = pd.read_csv(filePath, sep='\t', header=None)
    dataSet = np.array(df)
    dataSet = dataSet.astype(np.float32)

    return dataSet.tolist()


def loadDataSet1(fileName):
    """导入数据"""
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 使用python3会报错1，因为python3中map的返回类型是‘map’类，不能进行计算，需要将map转换为list
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def VecDistance(vecA, vecB):
    return math.sqrt(np.sum(np.power(vecA-vecB, 2)))


def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))

    for j in range(k):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j])-minJ)
        centroids[:, j] = minJ + rangeJ*np.random.rand(k, 1)    # 生成K个质心的坐标
    return centroids


def Kmeans(dataSet, k):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))

    centRoids = randCent(dataSet, k)    # 随机创建质心
    clusterChanged = True

    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = math.inf
            minIndex = -1
            for j in range(k):    # 对于每个点对其遍历每质心
                distJI = VecDistance(centRoids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:    # 如果点的质心改变就继续进行
                clusterChanged =True
            clusterAssment[i, :] = minIndex, minDist**2    # 记录点i最近的质心j
        print(centRoids)
        for cent in range(k):    # 更新质心位置
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]    # 把具有相同质心的点全部拿出来
            centRoids[cent, :] = np.mean(ptsInClust, axis=0)    # 矩阵.A是把矩阵转换为数组numpy,把相同质心的点求平均距离，纵向求和
    return centRoids, clusterAssment


if __name__ == '__main__':

    dataSet = loadDataSet('ex00.txt')
    centRoids, clusterAssment = Kmeans(np.array(dataSet), 2)
    print("质心为：", centRoids)
    print(clusterAssment)
