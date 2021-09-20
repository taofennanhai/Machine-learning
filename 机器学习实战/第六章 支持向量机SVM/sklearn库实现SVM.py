from sklearn import svm

X = [[2, 0], [1, 1], [2, 3]]
y = ['yes', 'yes', 'no']

clf = svm.SVC(kernel='linear')    # .SVC（）就是 SVM 的方程，参数 kernel 为线性核函数

clf.fit(X, y)    # 训练分类器

print(clf)    # 调用分类器的 fit 函数建立模型（即计算出划分超平面，且所有相关属性都保存在了分类器 cls 里）
print(clf.support_vectors_)    # 支持向量
print(clf.support_)    # 属于支持向量的点的 index
print(clf.n_support_)    # 在每一个类中有多少个点属于支持向量


print(clf.predict([[0, 0]]))    # 预测类别


# 获得划分超平面
# 划分超平面原方程：w0x0 + w1x1 + b = 0
# 将其转化为点斜式方程，并把 x0 看作 x，x1 看作 y，b 看作 w2
# 点斜式：y = -(w0/w1)x - (w2/w1)

w = clf.coef_[0]  # w 是一个二维数据，coef 就是 w = [w0,w1]
a = -w[0] / w[1]  # 斜率
# .intercept[0] 获得 bias，即 b 的值，b / w[1] 是截距
b = -(clf.intercept_[0]) / w[1]  # 带入 x 的值，获得直线方程
print("clf.coef_: ", clf.coef_)



