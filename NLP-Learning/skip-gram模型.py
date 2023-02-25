import torch
import torch.nn as nn
import torch.nn.functional as F

# 准备数据集
corpus = ['he is a king',
          'she is a queen',
          'he is a man',
          'she is a woman',
          'warsaw is poland capital',
          'berlin is germany capital',
          'paris is france capital']
corpus_list = [sentence.split() for sentence in corpus]

# 构建词典
word2ix = {}
for sentence in corpus:
    for word in sentence.split():
        if word not in word2ix:
            word2ix[word] = len(word2ix)    # 为每个词都匹配一个索引
ix2word = {v: k for k, v in word2ix.items()}    # 将dict中的key与value互换位置
voc_size = len(word2ix)

# 构建训练对
WINDOWS = 2  # 取左右窗口的词作为context_word
pairs = []  # 存放训练对

for sentence in corpus_list:    # 获取中心词和背景词的配对
    for center_word_index in range(len(sentence)):
        center_word_ix = word2ix[sentence[center_word_index]]
        for win in range(-WINDOWS, WINDOWS+1):
            contenx_word_index = center_word_index + win
            if 0 <= contenx_word_index <= len(sentence)-1 and contenx_word_index != center_word_index:
                context_word_ix = word2ix[sentence[contenx_word_index]]
                pairs.append((center_word_ix, context_word_ix))


print(pairs)


class SkipGramModel(nn.Module):
    def __init__(self, voc_dim, emb_dim):    # 第一个参数是所有词的个数，第二个是嵌入矩阵维数，例如300维
        super(SkipGramModel, self).__init__()

        self.embedding_matrix = nn.Parameter(torch.FloatTensor(emb_dim, voc_dim))    # 第一个设置的是中心词矩阵
        self.W = nn.Parameter(torch.FloatTensor(voc_dim, emb_dim))     # 第二个设置背景词矩阵

        torch.nn.init.xavier_normal_(self.embedding_matrix)
        torch.nn.init.xavier_normal_(self.W)

    def forward(self, x):
        emb = torch.matmul(self.embedding_matrix, x)    # 矩阵乘以one hot向量等于d*1维
        h = torch.matmul(self.W, emb)    # [voc_dim] 再用矩阵v*d乘以d*1维向量
        log_softmax = F.log_softmax(h, dim=0)  # [voc_dim] 最后使用softmax归一化得出概率，并且取log

        return log_softmax


# 提前设置超参数
epoch = 10
lr = 1e-2
embedding_dim = 5

model = SkipGramModel(voc_size, embedding_dim)
optim = torch.optim.Adam(model.parameters(), lr=0.01)
loss_f = torch.nn.NLLLoss()


# 这是将索引变成词典大小的One-Hot向量的方法
def get_onehot_vector(ix):
    one_hot_vec = torch.zeros(voc_size).float()
    one_hot_vec[ix] = 1.0
    return one_hot_vec


for e in range(epoch):
    epoch_loss = 0.0

    for i, (center_ix, context_ix) in enumerate(pairs):
        optim.zero_grad()

        one_hot_vector = get_onehot_vector(center_ix)
        y_true = torch.Tensor([context_ix]).long()

        # 前向传播 NLLLoss的结果就是把输出与Label对应的那个值拿出来，再去掉负号，再求均值。
        y_pred = model.forward(one_hot_vector)

        print((y_pred.view(1, -1)))
        loss = loss_f(y_pred.view(1, -1), y_true)    # 这里-1代表tensor大小，这里指的是15

        # 后向
        loss.backward()
        epoch_loss += loss.data.item()

        # 梯度更新
        optim.step()

        if e % 2 == 0:
            print('epoch: %d, loss: %f' % (e, epoch_loss))


# # 3.预测：预测单词的向量并计算相似度
v1 = torch.matmul(model.embedding_matrix, get_onehot_vector((word2ix['he'])))
v2 = torch.matmul(model.embedding_matrix, get_onehot_vector((word2ix['she'])))
v3 = torch.matmul(model.embedding_matrix, get_onehot_vector((word2ix['capital'])))

print(v1)
print(v2)
print(v3)

s_v1_v2 = F.cosine_similarity(v1, v2, dim=0)
s_v1_v3 = F.cosine_similarity(v1, v3, dim=0)
print(s_v1_v2)
print(s_v1_v3)





































