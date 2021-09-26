import numpy as np

def regLeaf(dataSet):
    """生成叶子节点，即目标变量的均值"""
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    """计算数据集中目标变量的误差平方和
    误差平方和 = 目标变量的均方差 * 数据集的样本个数
    """
    return np.var(dataSet[:, -1]) * dataSet.shape[0]

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


def binSplitDataSet(dataSet, feature, value):
    """
        通过数组过滤切分数据集
        :param dataSet: 数据集合
        :param feature: 待切分的特征
        :param value: 该特征的某个值
        :return:
        """
    # 使用python3会报错2，需要将书中脚本修改为以下内容
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1



def chooseBestSplit(dataSet, leafType, errType, ops):
    """
       遍历所有的特征及其可能的取值来找到使误差平方和最小化的切分特征及其切分点
       :param dataSet: 数据集合
       :param leafType: 建立叶节点的函数，该参数也决定了要建立的是模型树还是回归树
       :param errType: 代表误差计算函数,即误差平方和计算函数
       :param ops: 用于控制函数的停止时机，第一个是容许的误差下降值，第二个是切分的最少样本数
       :return:最佳切分特征及其切分点
    """
    tolS = ops[0]
    tolN = ops[1]

    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:    # 如果所有值都相等，则停止切分，直接生成叶结点
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)

    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal

    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)  # 用最佳切分特征和切分点进行切分
    if mat0.shape[0] < tolN or mat1.shape[0] < tolN:  # 如果切分出的数据集很小，则停止切分，直接生成叶结点
        return None, leafType(dataSet)

    return bestIndex, bestValue  # 返回最佳切分特征编号和切分点


def createCartTree(dataSet, leafType = regLeaf, errType = regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat    # 最佳切分特征
    retTree['spVal'] = val    # 切分点
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createCartTree(lSet, leafType, errType, ops)
    retTree['right'] = createCartTree(rSet, leafType, errType, ops)
    return retTree




if __name__ == '__main__':
    dataMat = loadDataSet('ex0.txt')
    dataMat = np.mat(dataMat)
    regTree = createCartTree(dataMat)
    print(regTree)





