import random
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import torch
import re
import 贝叶斯分类器 as bayesClassify
import os
import jieba


def textProcessing(folderPath, testSize = 0.2):
    folderList = os.listdir(folderPath)    # 读取文件价下面的目录
    dataList = []    # 一个类别中的数据
    classList = []   # 标记是否是垃圾新闻

    for folder in folderList:
        newFolderPath = os.path.join(folderPath, folder)    # 生成新的路径
        files = os.listdir(newFolderPath)    # 每个类别下的txt文件列表

        j = 1
        for file in files:
            if j > 100:    # 样本数如果超过100，就不需要了
                break
            with open(os.path.join(newFolderPath, file), 'r', encoding='utf-8') as f:    # 打开txt文件
                raw = f.read()

            wordCut = jieba.cut(raw)    # 文章切分
            wordList = list(wordCut)    # 切出来的数据转为list集合

            dataList.append(wordList)    # 添加数据集数据
            classList.append(folder)    # 给定数据类别

            j += 1

    dataClassList = list(zip(dataList, classList))    # 把数据和标签一起压缩
    random.shuffle(dataClassList)    # 随机打乱
    index = int(len(dataClassList) * testSize) + 1    # 训练集和数据集切分
    trainList = dataClassList[0:len(dataClassList) - index - 1]    # 前面均是训练集
    testList = dataClassList[len(dataClassList) - index - 1:]    # 后面均是测试集

    trainDataList, trainClassList = zip(*trainList)    # 解压缩
    testDataList, testClassList = zip(*testList)

    allWordsDict = {}    # 统计训练集的词频
    for wordsList in trainDataList:
        for word in wordsList:
            if word in allWordsDict.keys():
                allWordsDict[word] += 1
            else:
                allWordsDict[word] = 1

    allWordsTupleList = sorted(allWordsDict.items(), key=lambda f:f[1], reverse=True)
    allWordsList, allWordsNum = zip(*allWordsTupleList)
    allWordsList = list(allWordsList)    # 转换列表
    return allWordsList, trainDataList, testDataList, trainClassList, testClassList


def MakeWordsSet(words_file):
    words_set = set()                                            # 创建set集合
    with open(words_file, 'r', encoding='utf-8') as f:        # 打开文件
        for line in f.readlines():                                # 一行一行读取
            word = line.strip()                                    # 去回车
            if len(word) > 0:                                    # 有文本，则添加到words_set中
                words_set.add(word)
    return words_set                                             # 返回处理结果


def word_dict(all_words_list, deleteN, stopwords_set=set()):
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:
            break
        # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):                        # 出现在特征集中，则置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list                # 返回结果


def TextClassify(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)    # 调用函数
    test_score = classifier.score(test_feature_list, test_class_list)

    return test_score




if __name__ == '__main__':
    folderPath = 'SogouC/Sample'    # 存放训练集文件地址

    all_word_list, train_data_list, test_data_list, train_class_list, test_class_list = textProcessing(folderPath, 0.2)
    print(all_word_list)

    stopwords_file_path = 'stopwords_cn.txt'    # 停用词，太多无效的高频次会被去除
    setwords_set = MakeWordsSet(stopwords_file_path)

    test_accuracy_list = []
    deleteNs = range(0, 1000, 20)
    for deleteN in deleteNs:
        feature_words = word_dict(all_word_list, deleteN, setwords_set)    # 提取特征值
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)    # 获得训练样本的特征值
        test_accuracy = TextClassify(train_feature_list, test_feature_list, train_class_list, test_class_list)    # 进行分类
        test_accuracy_list.append(test_accuracy)    # 提取不同的特征值

    print(test_accuracy_list)



