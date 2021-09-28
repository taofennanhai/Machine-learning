import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def PCA(dataSet, topFeature=999):
    meanVals = np.mean(dataSet, axis=0)
    meanRemoved = dataSet - meanVals    # 去中心化
    covMat = np.cov(meanRemoved, rowvar=0)    # 计算协方差矩阵，其中rowvar=0，说明传入的数据一行代表一个样本；若非0，说明传入的数据一列代表一个样本。

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))    # 计算协方差矩阵的特征值和特征向量
    eigvalInd = np.argsort(eigVals)    # 返回特征值从小到大的下表
    eigvalInd = eigvalInd[:-(topFeature+1):-1]

    redEigVects = eigVects[:, eigvalInd]
    lowDDataMat = meanRemoved * redEigVects    # 数据转换成新空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals    # 降维数据进行重构

    return lowDDataMat, reconMat


def loadDataSet(filePath):

    df = pd.read_csv(filePath, sep='\t', header=None)
    dataSet = np.array(df)
    dataSet = dataSet.astype(np.float32)
    return dataSet.tolist()


if __name__ == '__main__':
    dataSet = loadDataSet('ex00.txt')
    lowDMat, reconMat = PCA(dataSet, 1)

    print(np.array(dataSet))

    plt.scatter(np.transpose(np.array(dataSet)[:, 0]), np.transpose(np.array(dataSet)[:, 1]), c='blue')
    plt.scatter(np.transpose(np.array(reconMat)[:, 0]), np.transpose(np.array(reconMat)[:, 1]), c='red', s=20, alpha=0.5)
    plt.show()







