import torch
import numpy as np

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, 0:8])
y_data = torch.from_numpy(xy[:, 8:9])

print(y_data)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()    #调用父类的构造函数，
        self.linear1 = torch.nn.Linear(8, 6)    #in_features：前一层网络神经元的个数 out_features： 该网络层神经元的个数以上两者决定了weight的形状[out_features , in_features]
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()    #Sigmoid模块做计算图
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        O1 = self.relu(self.linear1(x))
        O2 = self.relu(self.linear2(O1))
        y_pred = self.sigmoid(self.linear3(O2))
        return y_pred


model = Model()
criterion = torch.nn.BCELoss(reduction='mean')    #需要y_hat和y
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
