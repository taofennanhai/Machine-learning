import math

import sklearn.ensemble
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMat)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMat[:, dimen] <= threshVal] = -1.0         # 如果小于阈值,则赋值为-1
    else:
        retArray[dataMat[:, dimen] > threshVal] = -1.0    # 如果大于阈值,则赋值为-1
    return retArray


def buildStump(dataArr, classLabels, D):    # 和上面的函数共同构成单层决策树分类函数。本函数找到数据集上的最佳单层决策数

    dataMat = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = float('inf')    # 初始化最小误差为正无穷
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps    # 计算步长
        for j in range(-1, int(numSteps) + 1):    # 从-1开始切分
            for inequal in ['lt', 'gt']:    # lt代表小于该阈值分类为-1，gt代表大于该阈值为-1,因为有两种不同的结果
                threshVal = (rangeMin + float(j) * stepSize)    # 计算阈值
                predictedVal = stumpClassify(dataMat, i, threshVal, inequal)    # 计算分类结果
                errArr = np.mat(np.ones((m, 1)))    # 初始化误差矩阵
                errArr[predictedVal == labelMat] = 0    # 正确分类为0

                weightError = D.T * errArr    # 这里计算误差，比方说0.1 * 3
                if weightError < minError:
                    minError = weightError
                    bestClasEst = predictedVal.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainsDS(dataArr, classLabels, numIt):    # 训练弱分类器，其中numIt为最大迭代次数
    weakClassArr = []
    m = np.shape(dataArr)[0]    # 获得样本
    D = np.mat(np.ones((m, 1))/m)    # 每个样本的权重向量
    aggClassEst = np.mat(np.zeros((m, 1)))    # 类别累计估计值，就是f_m = Σα*G_m

    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)    # 构建单层决策树, classEst是决策树的估计值
        print("D:", D.T)

        alpha = float(0.5 * math.log2((1 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha    # 存储弱学习算法权重
        weakClassArr.append(bestStump)    # 存储单层决策树
        print('ClassEst:', classEst)

        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)    # 计算e的指数项目
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()

        aggClassEst += alpha * classEst    # 计算当前分类器下的类别的估计值

        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))    # 对每个样本分错的误差
        errorRate = aggErrors.sum() / m    # 误差率

        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    """
    AdaBoost分类函数
    Parameters:
        datToClass - 待分类样例
        classifierArr - 训练好的分类器
    Returns:
        分类结果
    """
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):                                        # 遍历所有分类器，进行分类
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


if __name__ == '__main__':
    datMat = np.matrix([[1., 2.1],
                        [1.5, 1.6],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    weakClassArr, aggClassEst = adaBoostTrainsDS(datMat, classLabels, 40)
    print(weakClassArr)
    print(aggClassEst)

    print(adaClassify([[1, 0], [5, 5]], weakClassArr))