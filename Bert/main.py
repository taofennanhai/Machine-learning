import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# BERT Parameters
from Bert import Bert

maxlen = 30    # 每句话最大的长度
batch_size = 4   # batch是每批6个
max_pred = 5  # max tokens of prediction
n_layers = 6
n_heads = 8
d_model = 512
d_ff = 512 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2  # 从序列中挑出两个句子

text = (
    'Hello, how are you? I am Romeo.\n'  # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'  # J
    'Nice meet you too. How are you today?\n'  # R
    'Great. My baseball team won the competition.\n'  # J
    'Oh Congratulations, Juliet\n'  # R
    'Thank you Romeo\n'  # J
    'Where are you going today?\n'  # R
    'I am going shopping. What about you?\n'  # J
    'I am going to visit my grandmother. she is not very well'  # R
)
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'

word_list = list(set(" ".join(sentences).split()))  # ['hello', 'how', 'are', 'you',...]
word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

for i, w in enumerate(word_list):
    word2idx[w] = i + 4
idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)

token_list = list()
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)


# sample IsNext and NotNext to be same in small batch size
def make_data():
    batch = []
    positive = negative = 0    # positive代表两个句子相邻 negative代表两个句子不相邻 保证采样的数据比例是1：1
    while positive != batch_size / 2 or negative != batch_size / 2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))  # 随机采样两句话的索引
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]    # 通过索引获取句子a,b

        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]    # 通过第一个特殊字符cls 和 sep 的拼接a b
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)    # 生成两个句子的位置编码 第一个cls符号是0 第一个sep是0 第二个 sep为1

        # MASK LM
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  # 取一个句子里面的15%的单词，max函数代表不足1取1，min函数代表超过最大预测数量去设定的最大预测数
        cand_maked_pos = [i for i, token in enumerate(input_ids) if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]  # 两个合并句子候选mask位置， 不包括cls sep

        shuffle(cand_maked_pos)    # 打乱

        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:    # 打乱之后取n_pred个位置做到随机mask效果
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])    # 把mask的真实单词编码放入
            # 这其中0.8到0.9的概率不做变化，所以不需要改变
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]']  # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1)  # random index in vocabulary
                while index < 4:  # 替换的时候不能包括 'CLS', 'SEP', 'PAD'
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index  # 用词库中的某个单词进行替换

        # Zero Paddings
        n_pad = maxlen - len(input_ids)    # 两句话组合不够最大长度，使用pad进行填充
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)    # input和segment也需要补

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:    # 保证mask数量就是5个，如果不够需要补0或者补1
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)    # mask和pos都需要补0或者1

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:    # 相邻的两个句子就是positive
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
            negative += 1
    return batch


# Proprecessing Finished

batch = make_data()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
    torch.LongTensor(input_ids), torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), \
    torch.LongTensor(masked_pos), torch.LongTensor(isNext)


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.isNext = isNext

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[idx]


loader = Data.DataLoader(MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext), batch_size, True)

model = Bert()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:

        # logit_lm是[batch_size, max_len, vocab_size]，需要进行前面几个预测
        # logits_clsf是[batch_size, n_segments]，预测是否为下一个句子
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)

        logits_lm = logits_lm.view(-1, vocab_size)    # 把logits_lm拉成[batch_size*max_len, vocab_size]，如[20,40]
        masked_tokens = masked_tokens.view(-1)    # 把masked_tokens拉成一条直线如[20,1],然后进行交叉熵运算

        # 首先进行每层softmax运算，按照每个对应预测点进行交叉熵运算如[25 30],就分别选中logit_lm中的第一行第25进行log(p)概率加上第二行第40进行log(p)，相加求和再取平均
        loss_lm = criterion(logits_lm, masked_tokens)    # masked language model 计算过程
        loss_lm = (loss_lm.float()).mean()    # 计算平均每个预测的loss均值

        loss_clsf = criterion(logits_clsf, isNext)    # for sentence classification
        loss = loss_lm + loss_clsf

        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Predict mask tokens ans isNext
input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[1]
print(text)
print('================================')
print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])

logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ', [pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list : ', [pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ', True if logits_clsf else False)
