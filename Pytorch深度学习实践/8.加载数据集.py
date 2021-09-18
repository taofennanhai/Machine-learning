import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


# 先是把数据导入x_data,y_data中
class DiabetesDataset(Dataset):    # Dataset是抽象类，必须重写方法
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]    # 知道xy是 N*9列，也就是xy的个数
        self.x_data = torch.from_numpy(xy[:, 0:8])
        self.y_data = torch.from_numpy(xy[:, 8:9])

    def __getitem__(self, index):    # 得到item下表
        return self.x_data[index], self.y_data[index]

    def __len__(self):    # 求len
        return self.len


dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=1)    # 设置batch-size大小，随机打乱，多线程输入数据


# 建立神经网络模型，输入层8个向量，第二层6个，第三层4个，输出层1个
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()    # 调用父类的构造函数，
        self.linear1 = torch.nn.Linear(8, 6)    # in_features：前一层网络神经元的个数 out_features： 该网络层神经元的个数以上两者决定了weight的形状[out_features , in_features]
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()    # Sigmoid模块做计算图
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        O1 = self.relu(self.linear1(x))
        O2 = self.relu(self.linear2(O1))
        y_pred = self.sigmoid(self.linear3(O2))
        return y_pred


model = Model()

# 构造使用模型，这里使用的是交叉熵y=-1/N*(y*log(y_hat)+(1-y)log(1-y_hat))
criterion = torch.nn.BCELoss(reduction='mean')  # 需要y_hat和y
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epoch_list = []
loss_list = []


# 训练数据，前向传播，反向传播，更新权重
def train(epoch):
    mean_loss =0.0
    for i, data in enumerate(train_loader, 0):  # 从0开始，循环总样本数/32次
        input, labels = data  # 每个input都是32*8列,label是32*1列
        y_pred = model(input)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())

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


if __name__ == '__main__':

    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    for epoch in range(10000):
        train(epoch)

    # train_dataset = datasets.MNIST(root='../dataset/mnist',
    #                                train=True,
    #                                transform=transforms.ToTensor,
    #                                download=True)
    # test_dataset = datasets.MNIST(root='../dataset/mnist',
    #                                train=False,
    #                                transform=transforms.ToTensor,
    #                                download=True)
    # trains_loader = DataLoader(dataset=train_dataset,
    #                            batch_size=32,
    #                            shuffle=True)
    # test_loader = DataLoader(dataset=test_dataset,
    #                          batch_size=32,
    #                          shuffle=False)


