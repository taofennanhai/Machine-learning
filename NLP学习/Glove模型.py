import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import sys
from collections import Counter
import scipy.spatial
from sklearn.metrics.pairwise import cosine_similarity

with open("train_1.txt", "r", encoding="utf-8") as f:    # 数据的读取
    corpus = f.read()

corpus = corpus.lower().split(" ")
vocab_size = 5000    # 设置前1w的单词

word_count = dict(Counter(corpus).most_common(vocab_size-1))     # 统计词数
word_count["<unk>"] = len(corpus) - sum(list(word_count.values()))    # 剩下的词标记为UNK,并且统计词数

idx2word = list(word_count.keys())    # 基本操作必须会，建立词索引
word2idx = {w: i for i, w in enumerate(word_count)}    # 反向索引


# 建立共现矩阵
window_size = 5    # 窗口大小
coMatrix = np.zeros((vocab_size, vocab_size), np.int32)
print("共现矩阵占内存大小为：{}MB".format(sys.getsizeof(coMatrix)/(1024*1024)))

corpus_encode = [word2idx.get(w, word2idx["<unk>"]) for w in corpus]    # 对语料库进行编码

for i, center_word in enumerate(corpus_encode):    # 进行共现矩阵的实现
    pos_indices = list(range(i-window_size, i)) + list(range(i+1, i+window_size+1))    # 中心窗口的选取
    temp = []
    for idx in pos_indices:
        if 0 < idx+i < len(corpus_encode):
            temp.append(idx)
    pos_indices = temp    # 防止下表越界
    context_word = [corpus_encode[idx] for idx in pos_indices]    # context_word代表当前center_word周围的所有关联词
    for word_idx in context_word:
        coMatrix[center_word][word_idx] += 1
    if (i + 1) % 1000000 == 0:
        print("已完成{}/{}".format(i + 1, len(corpus_encode)))

# # 统计出来的应该是对称矩阵
# for i in range(vocab_size):
#     # 遍历下三角即可
#     for j in range(i):
#         # 若不相等，则说明有错
#         if coMatrix[i][j] != coMatrix[j][i]:
#             print("共现矩阵有误行{}列{}不等".format(i, j))

del corpus, corpus_encode

xMax = 100    # 设置f(Xik)系数参数
alpha = 0.75

weight_Matrix = np.zeros_like(coMatrix, np.float32)
print("惩罚系数矩阵矩阵占内存大小为：{}MB".format(sys.getsizeof(weight_Matrix)/(1024*1024)))

for i in range(vocab_size):    # 对矩阵进行遍历
    for j in range(vocab_size):
        if 0 < coMatrix[i][j] < xMax:    # 设置小于xMax的为系数的0.75指数
            weight_Matrix[i][j] = (coMatrix[i][j]/xMax)**0.75
        elif coMatrix[i][j] > xMax:    # 超过阈值设置1
            weight_Matrix[i][j] = 1

# 以下创建DataLoader
batch_size = 10
embed_size = 100
epochs = 10

train_set = []
for i in range(vocab_size):    # 这里训练使用共现次数非零的作为训练数据
    for j in range(vocab_size):
        if coMatrix[i][j] != 0:
            train_set.append([i, j])
    if (i + 1) % 1000 == 0:
        print("已完成{}/{}".format(i+1, vocab_size))


class TrainDataSet(Dataset):
    def __init__(self, coMatrix, weight_Matrix, train_set):
        super(TrainDataSet, self).__init__()
        self.coMatrix = coMatrix
        self.weight_Matrix = weight_Matrix
        self.train_set = train_set

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, idx):
        i, k = self.train_set[idx]    # 中心词和背景词
        x_ik = self.coMatrix[i][k]    # 共现频次
        w = self.weight_Matrix[i][k]  # 权重矩阵
        return i, k, x_ik, w


dataset = TrainDataSet(coMatrix, weight_Matrix, train_set)
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)    # 一次可以读取10个中心词和背景词，和共现次数，权重

del coMatrix, weight_Matrix


# 寻找附近的词，和Word2vec一样
def nearest_word(word, embedding_weights):
    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights], np.float32)
    return [idx2word[i] for i in cos_dis.argsort()[:10]]    # 找到前10个最相近词语


# 建立GLoVe模型
class GloVeModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(GloVeModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.W = torch.nn.Embedding(vocab_size, embed_size)    # 设置中心词矩阵

        self.B_v = torch.nn.Embedding(vocab_size, 1)    # 中心词偏置和背景词偏置
        self.B_u = torch.nn.Embedding(vocab_size, 1)

        # 随机初始化参数(这种初始化方式收敛更快)，embedding原来是默认（0,1）正态分布
        initrange = 0.5 / vocab_size
        self.W.weight.data.uniform_(-initrange, initrange)
        self.B_v.weight.data.uniform_(-initrange, initrange)
        self.B_u.weight.data.uniform_(-initrange, initrange)

    def forward(self, i, k, x_ik, w):    # 输入的batch_size个中心词，背景词，共现频数，惩罚系数

        # w_i[batch,embed_size] w_k[batch,embed_size] b_i[batch,1]  b_k[batch,1]
        # 这里要清楚torch.mul()是矩阵按位相乘，两个矩阵的shape要相同
        # torch.mm()则是矩阵的乘法
        x_ik = x_ik.float()
        similarity = torch.mul(self.W(i), self.W(k))
        similarity = torch.sum(similarity, dim=1)

        b_i = self.B_u(i).t()
        b_k = self.B_v(k).t()

        loss = similarity + b_i + b_k - torch.log(x_ik)
        loss = loss * loss * w * 0.5

        return loss.sum().mean()

    def get_embeding(self):
        return self.W.weight.data.cpu().numpy()


model = GloVeModel(vocab_size, embed_size)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.05)

for epoch in range(epochs):
    allLoss = 0
    temp = torch.zeros(10, 1).cuda()
    for step, (i, k, co_occur, weight) in enumerate(dataloader):

        optimizer.zero_grad()
        loss = model(i, k, co_occur, weight)

        loss.backward()
        optimizer.step()

        loss = loss.cpu()
        allLoss += loss.item()

        if (step + 1) % 2000 == 0:
            print("epoch:{}, iter:{}, loss:{}".format(epoch + 1, step + 1, allLoss / (step + 1)))

        if (step + 1) % 5000 == 0:
            print("nearest to one is{}".format(nearest_word("one", model.get_embeding())))
















