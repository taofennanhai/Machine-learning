import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# y = torch.Tensor([1.0, 0.0, 0.0])
# z = torch.Tensor([0.2, 0.1, -0.1])
# z = F.softmax(z, dim=0)
# print(z)

# y_pred = np.exp(z) / np.exp(z).sum()
# loss = (-y*np.log(y_pred)).sum()
# print(loss)

# y = torch.LongTensor([2, 0, 1])
# y_pred1 = torch.Tensor([[0.1, 0.2, 0.9],
#                         [1.1, 0.1, 0.2],
#                         [0.2, 2.1, 0.1]])
# y_pred2 = torch.Tensor([[0.8, 0.2, 0.3],
#                         [0.2, 0.3, 0.5],
#                         [0.2, 0.2, 0.5]])
#
# y_pred1 = torch.nn.functional.softmax(y_pred1, dim=1)    # 先对输出的一列进行softmax层让其在0-1内
# y_pred1 = torch.log(y_pred1)    # 接着对每列取log
# nllloss = torch.nn.NLLLoss()    # 使用NLLloss,会先对每列取负，对标签进行计算loss
# loss = nllloss(y_pred1, y)
# print(loss)
#
# y_pred1 = torch.Tensor([[0.1, 0.2, 0.9],
#                         [1.1, 0.1, 0.2],
#                         [0.2, 2.1, 0.1]])
# criterion = torch.nn.CrossEntropyLoss()    # CrossEntropyLoss就是把以上Softmax–Log–NLLLoss合并成一步
# loss1 = criterion(y_pred1, y)v
# print(loss1)

