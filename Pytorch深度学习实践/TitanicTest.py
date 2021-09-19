import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

torch.set_printoptions(threshold=np.inf)
train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')

print('训练数据集：', train.shape, '测试数据集', test.shape)
rowNum_train = train.shape[0]
print('多少行训练数据集：', train.shape[0])

y_data = np.array([train['Survived'].values]).T

total = train.append(test, ignore_index=True)    # 数据纵向堆叠
total.describe()    # 显示统计的数据

print(total.isnull().sum())

total['Age'] = total['Age'].fillna(total['Age'].mean())    # 年龄用平均数代替

print('中位数为：', total['Fare'].quantile(0.5), total['Fare'].median())

total['Fare'] = total['Fare'].fillna(total['Fare'].quantile(0.5))    # 费用用中位数

total['Embarked'] = total['Embarked'].fillna('S')    # Embarked用S众数填充

print(total.describe())

sex_map = {'male': 1, 'female': 0}    # 男女使用二进制码代替
total['Sex'] = total['Sex'].map(sex_map)

# embark_map = {'S': 1, 'C': 2, 'Q': 3}
# total['Embarked'] = total['Embarked'].map(embark_map)


total = total.join(pd.get_dummies(total['Embarked']))    # 把单字符类型映射到二进制形式
# embarkedDf = pd.DataFrame()
# embarkedDf = pd.get_dummies(total['Embarked'], prefix='Embarked')
# total = pd.concat([total, embarkedDf], axis=1)
total.drop('Embarked', axis=1, inplace=True)    # 删除Embarked列

PlassdDf = pd.DataFrame()
PlassdDf = pd.get_dummies(total['Pclass'], prefix='Pclass')
total = pd.concat([total, PlassdDf], axis=1)
total.drop('Pclass', axis=1, inplace=True)    # 删除Pclass列

total.drop('Cabin', axis=1, inplace=True)    # 删除Cabin列
total.drop('Name', axis=1, inplace=True)    # 删除Name列
total.drop('Ticket', axis=1, inplace=True)    # 删除Ticket列

# print(total.isnull().sum())
# print(total['Sex'].head())

train = total[0:891]
test = total[891:]

# 数据预处理
class TatanicDataset():
    def __init__(self):
        label = train[['Survived']]
        inputs = train[['Sex', 'Age', 'SibSp', 'Parch',
                        'Fare', 'C', 'Q', 'S',
                        'Pclass_1', 'Pclass_2', 'Pclass_3']]
        self.y_data = torch.from_numpy(label.values)
        self.x_data = torch.from_numpy(inputs.values)
        self.x_data = self.x_data.float()    # 不这么写会有数据类型的错误
        self.y_data = self.y_data.float()
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):  # 得到item下表
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # 求len
        return self.len

tatanic_dataset = TatanicDataset()
train_loader = DataLoader(dataset=tatanic_dataset, batch_size=32, shuffle=True, num_workers=2)    # 设置batch-size大小，随机打乱，多线程输入数据

class TitanicModel(torch.nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.linear1 = torch.nn.Linear(11, 7)
        self.linear2 = torch.nn.Linear(7, 4)
        self.linear3 = torch.nn.Linear(4, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output1 = self.relu(self.linear1(x))
        output2 = self.relu(self.linear2(output1))
        y_pred = self.sigmoid(self.linear3(output2))
        return y_pred

titanicmodel = TitanicModel()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(titanicmodel.parameters(), lr=0.01)
epoch_list = []
loss_list = []

def train(epoch):
    mean_loss = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        y_pred = titanicmodel(inputs)
        loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss += loss


    mean_loss /= i
    epoch_list.append(epoch)
    loss_list.append(mean_loss.item())

    plt.clf()
    plt.plot(epoch_list, loss_list, '-r')
    plt.pause(0.01)

def save(num, value, filepath):
    dataframe = pd.DataFrame({'PassengerId': num, 'Survived': value})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(filepath, index=False, sep=',')




    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    for epoch in range(100):
        train(epoch)

    # print('w= ', model.linear1.weight.shape)
    # print('b = ',model.linear1.bias.shape)#输出参数

    correct = pd.read_csv('titanic/gender_submission.csv')
    label = np.array([correct['Survived'].values]).T


    inputs = test[['Sex', 'Age', 'SibSp', 'Parch',
                            'Fare', 'C', 'Q', 'S',
                            'Pclass_1', 'Pclass_2', 'Pclass_3']]
    inputs = torch.from_numpy(inputs.values)
    inputs = inputs.float()
    y_pred = titanicmodel.forward(inputs)

    count = 0.0
    sums = 0.0

    test_value = []
    for y_hat, y in zip(y_pred, label):
        if y_hat < 0.5:
            y_hat = 0
            test_value.append(0)
        else:
            y_hat = 1
            test_value.append(1)

        if y_hat == y:
            count += 1
        sums += 1

    print('正确率为：', count/sums)

    num = []
    for i in range(892, 1310):
        num.append(i)
    save(num, test_value, 'neuralNetworkEstimate.csv')  # 输出预测数值