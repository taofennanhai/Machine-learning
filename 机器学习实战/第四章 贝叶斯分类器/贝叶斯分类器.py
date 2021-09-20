import torch
import numpy as np


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],    # 句子每个切分
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    # 对每个句子标签

    return postingList, classVec


def createVocabularyList(dataSet):
    vocabularySet = set([])     # 每个单词都作为一个特征,转化为词向量
    for document in dataSet:    # 取每个句子
        vocabularySet = vocabularySet | set(document)    # 并集运算，所有单词数据放入一个set
    sortedVocab = list(vocabularySet)
    sortedVocab.sort()
    return sortedVocab


def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)    # 创建一个数据样本词向量
    for word in inputSet:    # 遍历句子中的每个单词
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1    # 如果单词在向量中
        else:
            print('error')
    return returnVec


def trainNB0(trainMat, trainCategory):
    numTrainDocs = len(trainMat)    # 计算训练样本
    numWords = len(trainMat[0])    # 计算每个样本的词条数

    pAbusive = sum(trainCategory) / float(numTrainDocs)    # 计算侮辱类的概率
    p0Num = np.ones(numWords)    # 拉普拉斯平滑
    p1Num = np.ones(numWords)

    p0Denom = 2
    p1Denom = 2    # 分母为2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:    # 这里计算条件概率，统计属于侮辱类的概率数据，即P(w0|侮辱类),P(w1|侮辱类)
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]    # 统计非侮辱类的概率
            p0Denom += sum(trainMat[i])
    p1Vect = np.log(p1Num/p1Denom)    # 取对数，防止数据下溢
    p0Vect = np.log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)    # 两个向量之间对应元素相乘，对应logAB = logA +logB
    p0 = sum(vec2Classify*p0Vec) + np.log(1.0 - pClass1)    # 这里计算的是p(love|非侮辱类)，p(my|非侮辱类)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    vocabList = createVocabularyList(postingList)

    trainMat = []    # 每条句子的词向量矩阵
    for postingDoc in postingList:
        trainMat.append(setOfWord2Vec(vocabList, postingDoc))    # 对每条句子矩阵进行词向量化

    p0V, p1V, pAb = trainNB0(trainMat, classVec)

    print(p1V, p0V, pAb)

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWord2Vec(vocabList, testEntry))    # 获得测试样本词向量数据

    print(testEntry, classifyNB(thisDoc, p0V, p1V, pAb))
