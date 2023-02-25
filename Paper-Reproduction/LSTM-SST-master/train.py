import fitlog

# fitlog.commit(__file__)             # auto commit your codes
# fitlog.add_hyper_in_file (__file__) # record your hyperparameters

import torch
from tqdm import tqdm
# from torch.autograd import  Variable
import T_f
import model
from reader import Reader
import numpy as np
# import fitlog
import torch.nn.utils.rnn as rnn
import argparse
import os


def coloate_fn(t_data):
    """

    :param data:
    :return:
    """
    data_list = []
    score_list = []
    for each in t_data:
        a, b = each
        data_list.append(b)
        score_list.append(a)
    # score,t_data=t_data
    # print(type(t_data))
    # print(t_data)
    data_list.sort(key=lambda x: len(x), reverse=True)
    data_list = rnn.pad_sequence(data_list, batch_first=True, padding_value=0)
    return torch.tensor(score_list), data_list


parser = argparse.ArgumentParser()
best_acc = 0
step = 0
best_step = 0
parser.add_argument("--optim", type=int, default=0)
parser.add_argument("--learnrate", type=float, default=0.05)
parser.add_argument("--embedding", type=int, default=168)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--layer", type=int, default=3)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--org", type=int, default=1)
parser.add_argument("--batch", type=int, default=1024)
parser.add_argument("--cuda", type=int, default=3)
parser.add_argument("--momentum", type=float, default=0.0001)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
# fitlog.add_hyper(args)
print("create train reader")
train_reader = Reader("./data/SST-cla/sentiment-train", 56)
test_reader = Reader("./data/SST2/sentiment-test", 56)
print("create test reader")
dev_reader = Reader("./data/SST2/sentiment-dev", 56)
# print("max sentence length",train_reader.max_line)
# print(test_reader.max_line)
train_loader = torch.utils.data.DataLoader(train_reader, shuffle=True, batch_size=args.batch)  # ,collate_fn=coloate_fn)
test_loader = torch.utils.data.DataLoader(test_reader, shuffle=True, batch_size=25)  # ,collate_fn=coloate_fn)
dev_loader = torch.utils.data.DataLoader(dev_reader, shuffle=True, batch_size=25)  # ,collate_fn=coloate_fn)
print("building model")
if args.org == 0:
    model = T_f.Modle(5, args.layer, args.embedding, args.dropout)
else:
    model = model.Modle(56, 5, args.layer, args.embedding, args.dropout)
print("model finished")
model = model.cuda()
model.train()
loss = []
model.zero_grad()
learning_epoch = args.epoch
embedding_size = 300
loss_function = torch.nn.CrossEntropyLoss()
# fitlog.add_hyper(0.9,name='momentum')
if args.optim == 0:
    optimizer = torch.optim.Adam(model.parameters(), args.learnrate)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learnrate, weight_decay=args.momentum)
tr = open("train.txt", 'w')
file = open("tests.txt", 'w')
for i in range(learning_epoch):
    model.train()
    losss = []
    acc = 0
    total = 0
    for num, (score, data) in tqdm(enumerate(train_loader)):
        step += 1
        optimizer.zero_grad()
        model.train()
        score = score.cuda()
        data = data.cuda()
        batch_size = score.size(0)
        # print("score",score)
        # print("data",data)
        # score = score.cuda()
        # data = data.cuda()
        predict = model(data)
        # print("predict",predict.shape)
        # print("score,",score.shape)
        loss = loss_function(predict, score)
        # print("loss:{}".format(loss.requires_grad))
        # loss=loss.requires_grad_()
        # print("score:{}".format(score.requires_grad))
        # print("predict:{}".format(predict.requires_grad))
        loss.backward()
        optimizer.step()
        # for each in zip(score,predict):
        #     tr.write(str(each)+'\n')
        # print(loss.grad)
        losss.append(loss.item())
        # tmp=zip(score,torch.argmax(predict,-1))
        # print(tmp)
        acc += (score == torch.argmax(predict, -1)).cpu().sum().item()
        total += score.size(0)

    print("training epoch %s accuracy is %s loss is %s " % (str(i), str(acc / total), str(np.mean(losss))))
    # fitlog.add_metric(np.mean(losss),name="loss",step=i)
    # fitlog.add_metric(acc/total,name='acc',step=i)
    # if num %100==0:
    torch.save(model.state_dict(), "model.bin")
    losss2 = []
    acc2 = 0
    total2 = 0
    model.eval()
    with torch.no_grad():
        for num, (score, data) in tqdm(enumerate(dev_loader)):
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
        print("dev epoch %s accuracy is %s loss is %s " % (str(i), str(acc2 / total2), str(np.mean(losss2))))
        # fitlog.add_metric(acc2/total2,name="test_acc",step=i)
        if acc2 / total2 > best_acc:
            best_acc = acc2 / total2
            # fitlog.add_best_metric({"dev":best_acc})
            losss2 = []
            acc2 = 0
            total2 = 0
            for num, (score, data) in tqdm(enumerate(test_loader)):
                batch_size = score.size(0)
                embedding_size = 300
                score = score.cuda()
                data = data.cuda()
                predict = model(data)
                loss = loss_function(predict, score)
                losss2.append(loss.item())
                acc2 += (score == torch.argmax(predict, -1)).cpu().sum().item()
                total2 += score.size(0)
                # for each in zip(score,predict,data):
                #     file.write(str(each)+"\n")
            print("test epoch %s accuracy is %s loss is %s " % (str(i), str(acc2 / total2), str(np.mean(losss2))))
            # fitlog.add_best_metric({'test':acc2/total2})
#
# fitlog.finish()                     # finish the logging
