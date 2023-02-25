import torch
import torch
from tqdm import tqdm
from torch.autograd import  Variable
from T_f import Modle
from reader import Reader
import numpy as np

import os
from torchsummaryX import summary
from tqdm import tqdm
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print("create train reader")
train_reader=Reader("./data/SST2/train.txt",56)
test_reader=Reader("./data/SST2/test.txt",56)
print("create test reader")
dev_reader=Reader("./data/SST2/dev.txt",56)
print("max sentence length",train_reader.max_line)
print(test_reader.max_line)
train_loader=torch.utils.data.DataLoader(train_reader,shuffle=True,batch_size=25)
test_loader=torch.utils.data.DataLoader(test_reader,shuffle=False,batch_size=256)
dev_loader=torch.utils.data.DataLoader(dev_reader,shuffle=False,batch_size=256)
print("building model")
model=Modle(train_reader.max_line,5,2,120,0.12)
print("model finished")
model=model.cuda()
model.train()
loss=[]
learning_epoch=20
embedding_size = 300
loss_function=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),0.001)
losss2 = []
acc2 = 0
total2 = 0
model.load_state_dict(torch.load("model.bin"))
model.eval()
i=0
tr=open("train.txt",'w')
file = open("tests.txt", 'w')
with torch.no_grad():
    for num, (score, data) in tqdm(enumerate(test_loader)):
        batch_size = score.size(0)
        embedding_size = 300
        score = score.cuda()
        data = data.cuda()
        predict = model(data)
        # print(predict.shape)
        loss = loss_function(predict, score)
        losss2.append(loss.item())
        acc2 += (score == torch.argmax(predict, -1)).cpu().sum().item()
        total2 += score.size(0)
    print("test epoch %s accuracy is %s loss is %s " % (str(i), str(acc2 / total2), str(np.mean(losss2))))
    losss2 = []
    acc2 = 0
    total2 = 0
    for num, (score, data) in tqdm(enumerate(dev_loader)):
        batch_size = score.size(0)
        embedding_size = 300
        score = score.cuda()
        data = data.cuda()
        predict = model(data)
        loss = loss_function(predict, score)
        losss2.append(loss.item())
        acc2 += (score == torch.argmax(predict, -1)).cpu().sum().item()
        total2 += score.size(0)
        for each in zip(score,predict,data):
            file.write(str(each)+"\n")
    print("dev epoch %s accuracy is %s loss is %s " % (str(i), str(acc2 / total2), str(np.mean(losss2))))