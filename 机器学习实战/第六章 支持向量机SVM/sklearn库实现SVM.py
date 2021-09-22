import numpy as np
import pandas
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# X = [[2, 0], [1, 1], [2, 3]]
# y = ['yes', 'yes', 'no']
#
# clf = svm.SVC(kernel='linear')    # .SVC（）就是 SVM 的方程，参数 kernel 为线性核函数
#
# clf.fit(X, y)    # 训练分类器
#
# print(clf)    # 调用分类器的 fit 函数建立模型（即计算出划分超平面，且所有相关属性都保存在了分类器 cls 里）
# print(clf.support_vectors_)    # 支持向量
# print(clf.support_)    # 属于支持向量的点的 index
# print(clf.n_support_)    # 在每一个类中有多少个点属于支持向量
#
#
# print(clf.predict([[0, 0]]))    # 预测类别
#
#
# # 获得划分超平面
# # 划分超平面原方程：w0x0 + w1x1 + b = 0
# # 将其转化为点斜式方程，并把 x0 看作 x，x1 看作 y，b 看作 w2
# # 点斜式：y = -(w0/w1)x - (w2/w1)
#
# w = clf.coef_[0]  # w 是一个二维数据，coef 就是 w = [w0,w1]
# a = -w[0] / w[1]  # 斜率
# # .intercept[0] 获得 bias，即 b 的值，b / w[1] 是截距
# b = -(clf.intercept_[0]) / w[1]  # 带入 x 的值，获得直线方程
# print("clf.coef_: ", clf.coef_)


def loadDataSet(filePath):
    data_set = pandas.read_csv(filePath, sep='\t', header=None)

    x = data_set.iloc[:, [0, 1]].values
    y = data_set.iloc[:, -1].values

    data_mat = x.tolist()
    label_mat = y.tolist()
    return data_mat, label_mat


def showClassifer(dataMat, w, b, alphas):
    # 绘制样本点
    data_plus = []                                  # 正样本
    data_minus = []                                 # 负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)            # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)    # 负样本散点图
    # 绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1)
    a2 = float(a2)
    y1, y2 = (-b - a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    for alpha in alphas:
        plt.scatter(alpha[0], alpha[1], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')

    plt.show()


def predictTestDataSet(testDataMat, testLabelMat, w, b):


    count = 0.0
    y_predict = []
    for i, data in enumerate(testDataMat):

        y_hat = np.sign(data[0] * w[0] + data[1] * w[1] + b)
        y_predict.append(y_hat)

        if y_hat == labelMat[i]:
            count += 1

    return count / len(testDataMat), y_predict


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')

    clf = svm.SVC(kernel='linear')  # .SVC（）就是 SVM 的方程，参数 kernel 为线性核函数

    clf.fit(dataMat, labelMat)  # 训练分类器

    print(clf.support_vectors_)

    w1 = float(clf.coef_[0][0])
    w2 = float(clf.coef_[0][1])
    w = [w1, w2]
    b = clf.intercept_[0]

    showClassifer(dataMat, w, b, clf.support_vectors_)

    dataMat1, labelMat1 = loadDataSet('testSetRBF.txt')
    clf1 = svm.SVC(C=20, kernel='rbf')    # 非线性的SVM
    clf1.fit(dataMat1, labelMat1)

    testDataMat, testLabelMat = loadDataSet('testSetRBF2.txt')

    predictResult = clf1.predict(testDataMat)
    print(predictResult)
    print(testLabelMat)
    # w1 = float(clf1.coef_[0][0])
    # w2 = float(clf1.coef_[0][1])
    # w = [w1, w2]
    # b = clf1.intercept_[0]

    # result, y_hat = predictTestDataSet(testDataMat, testLabelMat, w, b)
    #
    #

    print(w, b)

    print(clf1.score(testDataMat, testLabelMat))

    count = 0.0
    for i in range(len(predictResult)):
        if predictResult[i] == testLabelMat[i]:
            count += 1

    print(count/len(predictResult))

