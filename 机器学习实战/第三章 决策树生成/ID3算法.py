import operator

import torch
import numpy as np
import math
import matplotlib.pyplot as plt


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],         # 数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']		# 分类属性(特征)
    return dataSet, labels


def calculateShannonEnt(dataSet):
    # 返回数据集的行数
    num = len(dataSet)
    # 保存每个标签label出现的次数
    labelCount = {}

    for feature in dataSet:    # 对每个特征进行计算
        # 获取当前一行的label，也就是yes or no
        currentLabel = feature[-1]
        # 如果label没放入统计次字典，就添加进去
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1    # 计数

    shannonEnt = 0.0

    # 计算香农熵
    for key in labelCount:
        prob = float(labelCount[key]) / num
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


# axis = 0代表第一列特征， value = 0代表这一列特征的数据
def spiltDataSet(dataSet, axis, value):
    restDataSet = []
    for featVec in dataSet:
        # 如果取到了这列值
        if featVec[axis] == value:
            # 消除这列axis=0这列特征
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            # 把除这列特征放入剩余的dataset中
            restDataSet.append(reduceFeatVec)
    return restDataSet


def chooseBestFeatureToSplit(dataSet):
    featureNum = len(dataSet[0])-1
    baseEntropy = calculateShannonEnt(dataSet)

    # 信息增益
    bestInfoGain = 0.0
    # 最优特征下标索引值
    bestFeature = -1
    # 遍历其他所有特征
    for i in range(featureNum):
        # 取每一行list对象中第i个元素
        featList = [example[i] for example in dataSet]
        # 把一个个对象的属性值放入set数组中,如年龄的青年，老年，中年
        uniqueVal = set(featList)
        # 获取经验条件熵
        newEntropy = 0.0

        for value in uniqueVal:    # 计算每个类别的信息增益，如计算青年
            # 划分类别的子集
            subDataset = spiltDataSet(dataSet, i, value)
            # 计算子集所占的概率
            prob = len(subDataset) / float(len(dataSet))
            # 计算经验条件熵
            newEntropy += prob * calculateShannonEnt(subDataset)

        # 信息增益
        infoGain = baseEntropy - newEntropy
        # 更新最好的特征
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 投票表决法选择最多的类别
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createJudgeTree(dataSet, label):
    # 取分类标签
    classList = [example[-1] for example in dataSet]
    # 如果classList中类别全是一个类型，那么就不需要分类，直接返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 选择最优特征的标签
    bestFeatLabel = label[bestFeat]

    # 根据标签生成树
    judgeTree = {bestFeatLabel: {}}
    # 删除最优标签
    del(label[bestFeat])
    # 得到最优特征的属性值
    featVals = [example[bestFeat] for example in dataSet]

    # 划分属性值
    uniqueVal = set(featVals)
    for value in uniqueVal:
        subLabels = label[:]
        judgeTree[bestFeatLabel][value] = createJudgeTree(spiltDataSet(dataSet, bestFeat, value), subLabels)

    return judgeTree


def classify(judgeTree, featureLabel, testVec):
    firstStr = next(iter(judgeTree))
    secondDict = judgeTree[firstStr]    # 获取下一个字典
    featIndex = featureLabel.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classlabel = classify(secondDict[key], featureLabel, testVec)
            else:
                classlabel = secondDict[key]
    return classlabel


if __name__ == '__main__':
    dataset, label = createDataSet()

    judgetree = createJudgeTree(dataset, label)
    print(judgetree)

    testVec = [0, 1]                                        #测试数据
    featLabels = ['有自己的房子', '有工作']
    result = classify(judgetree, featLabels, testVec)

    print(result)




