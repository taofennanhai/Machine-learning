import os
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


def read_imdb(data_dir, is_train):
    """读取IMDb评论数据集文本序列和标签"""
    data, labels = [], []
    for label in ('pos', 'neg'):    # 先读取pos的评论
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)    # 获取文件夹的名称
        for file in os.listdir(folder_name):    # 遍历文件夹中的文件
            with open(os.path.join(folder_name, file), 'rb') as f:    # 联合两个路径，打开文件
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


# train_data = read_imdb('../Bert/aclImdb', is_train=True)
# print('训练集数目：', len(train_data[0]))
#
# for x, y in zip(train_data[0][24997:24999], train_data[1][24997:24999]):
#     print('标签：', y, 'review:', x[0:60])
#
#
# train_tokens = d2l.tokenize(train_data[0], token='word')    # 把句子划分每个单词
# vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])    # 保留每个单词大于5的单词，并统计次数
#
#
# d2l.set_figsize()
# d2l.plt.xlabel('# tokens per review')
# d2l.plt.ylabel('count')
# d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50))    # 查看不同句子的长度
# d2l.plt.show()
#
#
# num_steps = 500  # 设置固定的序列长度，大于这个长度的句子裁剪，小于的删除
# train_features = torch.tensor([d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])    # 把单词序列化并裁剪为固定长度的tensor
# print(train_features.shape)
#
#
# train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)
#
# for X, y in train_iter:
#     print('X:', X.shape, ', y:', y.shape)
#     break
# print('小批量数目：', len(train_iter))


def load_imdb_data(batch_size, num_steps=500):
    train_data = read_imdb('../../Bert/aclImdb1', is_train=True)    # 获取训练的数据集，共有25000条数据
    test_data = read_imdb('../../Bert/aclImdb1', is_train=False)  # 获取测试的数据集

    train_tokens = d2l.tokenize(train_data[0], token='word')    # 把句子划分每个单词
    test_tokens = d2l.tokenize(test_data[0], token='word')

    vocab = d2l.Vocab(train_tokens, min_freq=5)    # 保留每个单词大于5的单词，并统计次数，获取token_to_index,和index_to_token

    train_features = torch.tensor([d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])    # 把单词序列化并裁剪为固定长度的tensor
    test_features = torch.tensor([d2l.truncate_pad(vocab[line], num_steps, vocab['pad']) for line in test_tokens])

    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])), batch_size, is_train=False)    # 加载测试数据部分

    return train_iter, test_iter, vocab


def get_train_test_dataset():
    train_dataset, train_label = read_imdb('../../Bert/aclImdb1', is_train=True)  # 获取训练的数据集，共有25000条数据
    test_dataset, test_label = read_imdb('../../Bert/aclImdb1', is_train=False)  # 获取测试的数据集

    return train_dataset, train_label, test_dataset, test_label

