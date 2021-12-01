import torch
import torch.utils.data as tud
import numpy as np
from collections import Counter
import scipy.spatial


import fasttext
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity

C = 2  # 周围单词个数
K = 10  # 负采样（number of negative samples）
vocab_size = 4000
embed_size = 50


def loadDataset(filename):  # 第一步先加载词
    with open(filename, "r", encoding="utf-8") as f:  # 数据的读取
        corpus = f.read()

    corpus = corpus.lower().split(" ")

    for i, word in enumerate(corpus):  # 把字符数小于3的单词进行数据填充
        if len(word) == 1:
            corpus[i] = word + "@@"
        elif len(word) == 2:
            corpus[i] = word + "@"

    word_count = dict(Counter(corpus).most_common(vocab_size - 1))  # 统计词数
    word_count["<unk>"] = len(corpus) - sum(list(word_count.values()))  # 剩下的词标记为UNK,并且统计词数

    idx2word = list(word_count.keys())  # 基本操作必须会，建立词索引
    word2idx = {w: i for i, w in enumerate(word_count)}  # 反向索引

    return corpus, idx2word, word2idx, word_count


def get_word_ngram(word, trigram2idx):  # 获取单词的3_gram的集合以及下表索引
    temp = []
    index = []
    for i in range(len(word)):
        if i == 0:
            temp.append("<" + word[i:i + 2])
            index.append(trigram2idx["<" + word[i:i + 2]])
        elif i == len(word) - 1:
            temp.append(word[i - 1:] + ">")
            index.append(trigram2idx[word[i - 1:] + ">"])
        else:
            temp.append(word[i - 1:i + 2])
            index.append(trigram2idx[word[i - 1:i + 2]])

    return temp, index


def n_gramSet(word2idx):  # 获取词典中每个单词的3-gram，和4-gram集合

    temp = []
    for wordtoidx in word2idx:  # 获取3-gram集合
        for i in range(len(wordtoidx)):
            if i == 0:
                temp.append("<" + wordtoidx[i:i + 2])
            elif i == len(wordtoidx) - 1:
                temp.append(wordtoidx[i - 1:] + ">")
            else:
                temp.append(wordtoidx[i - 1:i + 2])

    trigram_count = dict(Counter(temp).most_common())  # 统计词数
    idx2trigram = list(trigram_count.keys())  # 基本操作必须会，建立词索引
    trigram2idx = {w: i for i, w in enumerate(trigram_count)}  # 反向索引
    del temp

    return trigram2idx, idx2trigram


class FastText(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, ngram_vocab_size, idx2word, trigram2idx):
        super(FastText, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.idx2word = idx2word
        self.trigram2idx = trigram2idx

        initrange = 0.5 / self.embed_size
        self.center_embedding = torch.nn.Embedding(vocab_size, embed_size)  # 中心词矩阵
        self.background_embedding = torch.nn.Embedding(vocab_size, embed_size)  # 背景词矩阵

        self.trigram_embedding = torch.nn.Embedding(ngram_vocab_size, embed_size)  # 3-gram词向量

        self.center_embedding.weight.data.uniform_(-initrange, initrange)
        self.background_embedding.weight.data.uniform_(-initrange, initrange)
        self.trigram_embedding.weight.data.uniform_(-initrange, initrange)

        # self.embedding_fourgram = torch.nn.Embedding(ngram_vocab_size, embed_size,    # 4-gram集合词向量
        #                                       padding_idx=ngram_vocab_size - 1)

    def forward(self, input_labels, pos_labels, neg_labels):
        # input_labels:[batch_size]
        # pos_labels:[batch_size, (windows_size * 2)]
        # neg_labels:[batch_size,  K]

        input_embedding = self.center_embedding(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.background_embedding(pos_labels)  # [batch_size, (windows_size * 2), embed_size]
        neg_embedding = self.background_embedding(neg_labels)  # [batch_size,  K, embed_size]

        W_t = input_labels.numpy().tolist()
        Loss = 0
        for i,w_t in enumerate(W_t):
            temp, index = get_word_ngram(self.idx2word[w_t], self.trigram2idx)
            index = torch.tensor(index, dtype=torch.long)
            mat = self.trigram_embedding(index)

            mat = torch.cat((input_embedding[i].view(1, 50), mat), dim=0)

            pos_wt_wc = torch.sum(torch.mm(mat, pos_embedding[i].T), dim=0)    # 中心词n-gram集合与正样本乘积的求和
            pos_wt_wc = torch.log(1+torch.exp(-pos_wt_wc))

            neg_wt_wc = torch.sum(torch.mm(mat, neg_embedding[i].T), dim=0)    # 中心词n-gram集合与负样本乘积的求和
            neg_wt_wc = torch.log(1 + torch.exp(neg_wt_wc))

            Loss += pos_wt_wc.sum() + neg_wt_wc.sum()

        return Loss

    def get_embeding(self):
        return self.center_embedding.weight.data.cpu().numpy()


class DataSet(tud.Dataset):
    def __init__(self, corpus, word2idx, idx2word, word_freqs, word_counts):
        super(DataSet, self).__init__()
        self.corpus_encoded = [word2idx.get(word, word2idx["<unk>"]) for word in corpus]
        self.corpus_encoded = torch.Tensor(self.corpus_encoded).long()  # 解码为词表中的索引
        self.wrd2idx = word2idx  # 词：索引 的键值对
        self.idx2word = idx2word  # 词（列表）
        self.word_freqs = torch.Tensor(word_freqs)  # 词频率
        self.word_counts = torch.Tensor(word_counts)  # 词个数o

    def __len__(self):
        # 共有多少个item
        return len(self.corpus_encoded)

    def __getitem__(self, idx):
        # 根据idx返回
        center_word = self.corpus_encoded[idx]  # 找到中心词
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))  # 中心词前后各C个词作为正样本

        pos_indices = [i % len(self.corpus_encoded) for i in pos_indices]  # 取余,以防超过文档范围
        pos_words = self.corpus_encoded[pos_indices]  # 周围单词
        neg_words = torch.multinomial(self.word_freqs, K, replacement=False)

        return center_word, pos_words, neg_words


def nearest_word(word, embedding_weights, word2idx, idx2word):    # 寻找附近的词，和Word2vec一样
    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights], np.float32)

    return [idx2word[i] for i in cos_dis.argsort()[:10]]    # 找到前10个最相近词语


if __name__ == '__main__':

    # corpus, idx2word, word2idx, word_count = loadDataset("train_1.txt")
    #
    # word_counts = np.array([count for count in word_count.values()], dtype=np.float32)
    # word_freqs = word_counts / np.sum(word_counts)
    #
    # word_freqs = word_freqs ** (3. / 4.)
    # word_freqs = word_freqs / np.sum(word_freqs)
    #
    # # print(corpus)
    # print(word_count["<unk>"])
    # trigram2idx, idx2trigram = n_gramSet(word2idx)
    #
    # # trigram_set, index2trigram = get_word_ngram("a@@", trigram2idx)
    # # print(trigram_set, index2trigram)
    #
    # ngram_vocab_size = (len(trigram2idx))
    #
    # dataset = DataSet(corpus, word2idx, idx2word, word_freqs, word_counts)
    # dataloader = tud.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)
    #
    # fasttext = FastText(vocab_size, embed_size, ngram_vocab_size, idx2word, trigram2idx)
    # optim = torch.optim.Adagrad(fasttext.parameters(), lr=0.01)

    # for epoch in range(10):
    #     epoch_loss = 0
    #     for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
    #
    #         optim.zero_grad()
    #         loss = fasttext.forward(input_labels, pos_labels, neg_labels)
    #         loss.backward()
    #         epoch_loss += loss.data.item()
    #         # 梯度更新
    #         optim.step()
    #
    #         if (i + 1) % 1000 == 0:
    #             print("epoch:{}, iter:{}, loss:{}".format(epoch + 1, i + 1, epoch_loss / (i + 1)))
    #
    #         if (i + 1) % 2000 == 0:
    #             print("nearest to one is{}".format(nearest_word("english", fasttext.get_embeding(), word2idx, idx2word)))

    model = fasttext.train_unsupervised("train_1.txt", "skipgram")
    print([model.get_word_vector(x) for x in ["one", "two", "a"]])
    print("---------")
    print(model.get_nearest_neighbors("english"))
    print(model.get_analogies("berlin", "germany", "france"))
