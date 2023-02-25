import torch
import zipfile
import collections
import numpy as np
import random
import torch.utils.data as tud
import torch.nn.functional as F
import nltk
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
import pickle
from collections import Counter
from collections import defaultdict
import torch.optim as optim
from tqdm import tqdm

embedding_dim = 128    # 词嵌入维度
voc_size = 10000    # 词表大小
epoch = 5
window_size = 5    # 周边词窗口大小
neg_samples = 3    # 负样本大小
print_every = 100    # 可视化频率
freq = 5    # 词汇出现频数的阈值
delete_words = False    # 是否删除部分高频词


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        # 读取出来的每个单词是 bytes
        data = f.read(f.namelist()[0]).split()
        # 把 bytes 转换为 str
        data = [str(x, encoding="utf8") for x in data]
        # SecondQuestionData = list(map(lambda x: str(x, encoding="utf8"), SecondQuestionData))
    return data


words = read_data("text8.zip")
print('Data size', len(words))


counts_dict = dict(collections.Counter(words).most_common(voc_size-1))    # 取出频数前 10000 的单词

counts_dict['UNK'] = len(words)-np.sum(list(counts_dict.values()))


ix2word = [word for word in counts_dict.keys()]    # 每个词有独立的索引
word2ix = {word: i for i, word in enumerate(ix2word)}    # 建立反索引

data = [word2ix.get(word, word2ix["UNK"]) for word in words]    # 把单词列表转换为编号的列表

print(ix2word)

total_count = len(data)    # 计算词频
word_freqs = {w: c/total_count for w, c in counts_dict.items()}

if delete_words:    # 以一定概率去除出现频次高的词汇
    t = 1e-5
    prob_drop = {w: 1-np.sqrt(t/word_freqs[w]) for w in data}
    data = [w for w in data if random.random() < (1-prob_drop[w])]
else:
    data = data

word_counts = np.array([count for count in counts_dict.values()], dtype=np.float32)    # 计算词频,按照原论文转换为3/4次方
word_freqs = word_counts/np.sum(word_counts)    # 先计算词频

word_freqs = word_freqs ** 3/4    #
word_freqs = word_freqs/np.sum(word_freqs)    # 概率的计算


# DataLoader自动帮忙生成batch
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, data, word_freqs):
        super(WordEmbeddingDataset, self).__init__()
        self.data = torch.Tensor(data).long()  # 解码为词表中的索引
        self.word_freqs = torch.Tensor(word_freqs)  # 词频率

    def __len__(self):
        # 共有多少个item
        return len(self.data)

    def __getitem__(self, idx):
        # 根据idx返回
        center_word = self.data[idx]  # 找到中心词
        pos_indices = list(range(idx - window_size, idx)) + list(range(idx + 1, idx + window_size + 1))    # 中心词前后各C个词作为正样本
        pos_indices = list(filter(lambda i: i >= 0 and i < len(self.data), pos_indices))    # 过滤，如果索引超出范围，则丢弃
        pos_words = self.data[pos_indices]  # 周围单词
        # 根据 变换后的词频选择 K * 2 * C 个负样本，True 表示可重复采样
        neg_words = torch.multinomial(self.word_freqs, neg_samples * pos_words.shape[0], True)

        return center_word, pos_words, neg_words


class SkipGramModelWithNegSample(torch.nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGramModelWithNegSample, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size    # 权重初始化参数
        self.center_embedding = torch.nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.center_embedding.weight.data.uniform_(-initrange, initrange)  # 权重初始化的一种方法

        initrange = 0.5 / self.embed_size  # 权重初始化参数
        self.back_embedding = torch.nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.back_embedding.weight.data.uniform_(-initrange, initrange)  # 权重初始化的一种方法

    def forward(self, input_labels, pos_labels, neg_labels):
        # input_labels:[batch_size]
        # pos_labels:[batch_size, windows_size*2]
        # neg_labels:[batch_size, windows_size * N_SAMPLES]

        input_embedding = self.center_embedding(input_labels)
        pos_embedding = self.back_embedding(pos_labels)
        neg_embedding = self.back_embedding(neg_labels)

        # 向量乘法  
        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1],新增一个维度用于向量乘法
        # input_embedding = input_embedding.view(BATCH_SIZE, EMBEDDING_DIM, 1)
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze(2)  # [batch_size, windows_size * 2] 只保留前两维
        neg_dot = torch.bmm(neg_embedding.neg(), input_embedding).squeeze(2)  # [batch_size, windows_size * 2 * K] 只保留前两维

        log_pos = F.logsigmoid(pos_dot).sum(1)  # 按照公式计算
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = -(log_pos + log_neg)  # [batch_size]

        return loss

    def input_embeddings(self):
        # #取出self.in_embed数据参数
        return self.in_embed.weight.data.cpu().numpy()


# 构造  dataset 和 dataloader
dataset = WordEmbeddingDataset(data, word_freqs)
dataloader = tud.DataLoader(dataset, batch_size=64, shuffle=True)

# 定义一个模型
model = SkipGramModelWithNegSample(voc_size, embedding_dim)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 迭代
for epoch in range(10):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):

        input_labels = input_labels.long()  # 全部转为LongTensor
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()

        optimizer.zero_grad()  # 梯度归零
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("epoch", epoch, "loss", loss.item())







