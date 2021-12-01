import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas
import torchvision

df = pandas.read_csv('testSet.txt', sep='\t', header=None)
x = df.iloc[:, [0, 1]]
y = df.iloc[:, -1]
x_data = torch.from_numpy(x.values)
y_data = torch.from_numpy(y.values)
y_data = y_data.reshape(y_data.shape[0], 1)
x_data = x_data.float()
y_data = y_data.float()

def loadDataSet():
    dataMat = []  # 创建数据列表
    labelMat = []  # 创建标签列表
    fr = open('testSet.txt')  # 打开文件
    for line in fr.readlines():  # 逐行读取
        lineArr = line.strip().split()  # 去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 添加数据
        labelMat.append(int(lineArr[2]))  # 添加标签
    fr.close()  # 关闭文件
    return dataMat, labelMat  # 返回


def plotDataSet():
    dataMat, labelMat = loadDataSet()                                    # 加载数据集
    dataArr = np.array(dataMat)                                            # 转换成numpy的array数组
    n = np.shape(dataMat)[0]                                            # 数据个数
    xcord1 = []; ycord1 = []                                            # 正样本
    xcord2 = []; ycord2 = []                                            # 负样本
    for i in range(n):                                                    # 根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            # 添加subplot
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)    # 绘制正样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)            # 绘制负样本
    plt.title('DataSet')                                                # 绘制title
    plt.xlabel('x')
    plt.ylabel('y')                                    # 绘制label
    plt.show()


def plotBestFit(weights, bias):
    dataMat, labelMat = loadDataSet()                                    # 加载数据集
    dataArr = np.array(dataMat)                                            # 转换成numpy的array数组
    n = np.shape(dataMat)[0]                                            # 数据个数
    xcord1 = []; ycord1 = []                                            # 正样本
    xcord2 = []; ycord2 = []                                            # 负样本
    for i in range(n):                                                    # 根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])    # 1为正样本
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])    # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)    # 绘制正样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)            # 绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)

    y = weights[0]*x + weights[1]*x + bias
    ax.plot(x, y)
    plt.title('BestFit')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                    #绘制label
    plt.show()


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()    #调用父类的构造函数，
        self.linear = torch.nn.Linear(2, 1)    #in_features：前一层网络神经元的个数 out_features： 该网络层神经元的个数以上两者决定了weight的形状[out_features , in_features]

    def forward(self, x_test):
        y_pred = torch.sigmoid(self.linear(x_test))
        return y_pred


model = LogisticRegressionModel()
criterion = torch.nn.BCELoss(reduction='mean')    #需要y_hat和y
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(15000):    #第一步先算y_hat，第二部分算损失，第三步一就是反向传播，最后一步是梯度更新
    y_pred = model.forward(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

w = model.linear.weight.data
b = model.linear.bias.data

print(' w0=', w.numpy()[0][0], ' w1=', w.numpy()[0][1])
print(' b=', b.item())
plotDataSet()

weights = [w.numpy()[0][0], w.numpy()[0][1]]
plotBestFit(weights, b.item())










#
# x_test = torch.Tensor([[4.0]])
# y_test = model.forward(x_test)
# print('y_test:', y_test.item())



# pred = torch.tensor([[-0.2], [0.2], [0.8]])
# target = torch.tensor([[0.0], [0.0], [1.0]])
#
# sigmoid = torch.nn.Sigmoid()
# pred_s = sigmoid(pred)
# print(pred_s)
# """
# pred_s 输出tensor([[0.4502],[0.5498],[0.6900]])
# 0*math.log(0.4502)+1*math.log(1-0.4502)
# 0*math.log(0.5498)+1*math.log(1-0.5498)
# 1*math.log(0.6900) + 0*log(1-0.6900)
# """
# result = 0
# i = 0
# for label in target:
#     if label.item() == 0:
#         result += math.log(1 - pred_s[i].item())
#     else:
#         result += math.log(pred_s[i].item())
#     i += 1
# result /= 3
# print("bce：", -result)
# loss = torch.nn.BCELoss()
# print('BCELoss:', loss(pred_s, target).item())
