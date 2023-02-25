import os
import re
import torch
from torch import nn
from d2l import torch as d2l


def Read_Snli(is_train):    # 读取SNLI数据集 将数据分解为前提 假设 和标签

    def extract_text(s):    # 删除我们不会使用的信息
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 用一个空格替换两个或多个连续的空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = '../../dataset/SNLI/original/snli_1.0_dev.txt' if is_train else '../../dataset/SNLI/original/snli_1.0_test.txt'

    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
        premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
        hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
        labels = [label_set[row[0]] for row in rows if row[0] in label_set]

    return premises, hypotheses, labels


train_data = Read_Snli(True)

for premise, hypothesis, label in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('前提：', premise)
    print('假设：', hypothesis)
    print('假设：', label)


class SNLIDataSet(torch.utils.data.Dataset):    # 定义数据集类

    def __init__(self, dataset, num_steps, vocab=None):    # num_step是规定文本长度，大于长度截断，小于填补。 vocab是词汇表
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])    # 分别赋值给token词典

        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab

        self.premises = self._pad(all_premise_tokens)    # 建立映射关系
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):    # 给文本数据填充
        return torch.tensor([d2l.truncate_pad(self.vocab[line], self.num_steps, self.vocab['<pad>']) for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)


def load_data_snli(batch_size, num_steps=50):    # 下载SNLI数据集并返回数据迭代器和词表

    num_workers = 1    # 这里默认用四个线程调用读取
    train_data = Read_Snli(True)
    test_data = Read_Snli(False)

    train_set = SNLIDataSet(train_data, num_steps)
    test_set = SNLIDataSet(test_data, num_steps, train_set.vocab)    # 定义加载类，使用线程加载数据

    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size, shuffle=True, num_workers=num_workers)

    return train_iter, test_iter, train_set.vocab


train_iter, test_iter, vocab = load_data_snli(128, 50)    # 加载数据集
len(vocab)

for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
