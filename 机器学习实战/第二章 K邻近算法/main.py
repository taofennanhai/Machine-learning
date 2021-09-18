import operator

import numpy as np


def createDataSet():
    group = np.array([[1, 100], [5, 89], [108, 5], [115, 8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


# KNN算法核心
def classifly0(inX, dataSet, labels, k):
    # 返回dataset的行数
    dataSetSize = dataSet.shape[0]
    # 把第inx向量纵向复制dataSize的大小，然后减去数据集
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 向量每个元素平凡
    sqDiffMat = diffMat**2
    # 每一行的列相加，获取距离每个数据集样本的大小
    sqDistances = sqDiffMat.sum(axis=1)
    # 向量每个元素开方
    distances = sqDistances ** 0.5
    # 返回distance下标索引
    sortedDistindicies = distances.argsort()

    classCount = {}
    for i in range(k):
        # 获取前k个最小的标签
        voteIlabel = labels[sortedDistindicies[i]]
        # 若不存在voteIlabel，则字典classCount中生成voteIlabel元素，并使其对应的数字为0，即
        # classCount = {voteIlabel：0}
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

        # python3中用items()替换python2中的iteritems()
        # key=operator.itemgetter(1)根据字典的值进行排序
        # key=operator.itemgetter(0)根据字典的键进行排序
        # reverse降序排序字典
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1),
                                  reverse=True)
        # 返回最多的类别，即分好了类
        return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = createDataSet()
    print(labels)
    classed = classifly0([0, 0], group, labels, 3)
    print(classed)


