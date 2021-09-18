import random

import numpy as np
import torch
import re
import 贝叶斯分类器 as bayesClassify


def textParse(bigString):
    listOfTokens = re.split('\\W+', bigString)    # 特殊字符作为切分标准，非数字非字符
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]    # 除去单个字符，并且把单词的首个字母转为小写字母


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个数据样本词向量
    for word in inputSet:  # 遍历句子中的每个单词
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1  # 如果单词在向量中
        else:
            print('error')
    return returnVec



if __name__ == '__main__':
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):    # 遍历每个垃圾邮件
        wordList = textParse(open('email/spam/%d.txt' % i).read())    # 读取垃圾邮件
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())  # 读取非垃圾邮件
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabularyList = bayesClassify.createVocabularyList(docList)    # 创建一个不重复的set集合

    trainSet = list(range(50))
    testSet = []    # 随机挑出数据集中作为测试集

    for i in range(10):    # 选出10条作为数据测试集，40条作为训练集
        randIndex = int(random.uniform(0, len(trainSet)))    # 随机选取索引值
        testSet.append(trainSet[randIndex])    # 选取测试集的索引值
        del(trainSet[randIndex])    # 删除所选的索引值,之后就没有测试集

    trainMat = []
    trainClass = []
    for docIndex in trainSet:
        trainMat.append(bagOfWords2VecMN(vocabularyList, docList[docIndex]))    # 获得每个训练集的词向量
        trainClass.append((classList[docIndex]))
    p0V, p1V, pSam = bayesClassify.trainNB0(np.array(trainMat), np.array(trainClass))    # 第一个是非垃圾邮件向量的条件概率，第二个是垃圾邮件向量的条件概率，第三个是垃圾邮件的概率
    errorCount = 0.0

    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabularyList, docList[docIndex])    # 把测试样例转化为词集模型
        if bayesClassify.classifyNB(np.array(wordVector), p0V, p1V, pSam) != classList[docIndex]:
            errorCount += 1

    print('error rate: ', errorCount / len(trainSet))









