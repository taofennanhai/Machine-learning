import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from Read_SNLI_DataSet import load_data_snli

def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:                                #假设类型为 torch.tensor 的张量 t 的形状如(2,4,3,5,6)，则 torch.flatten(t, 1, 3).shape 的结果为 (2, 60, 6)。
        net.append(nn.Flatten(start_dim=1))    # 将索引为start_dim 和end_dim之间（包括该位置）的数量相乘，其余位置不变

    net.append(nn.Dropout(0.2))      # 为啥要两次
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))

    return nn.Sequential(*net)    # 不带✳的是列表，带✳的是元素，所以nn.Sequential(*net[3: 5])中的*net[3: 5]就是给nn.Sequential()这个容器中传入多个层


class Attend(nn.Module):

    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):    # A/B的形状：（批量大小，序列A/B的词元数，embed_size）

        f_A = self.f(A)
        f_B = self.f(B)    # f_A/f_B的形状：（批量大小，序列A/B的词元数，num_hiddens）
                           # torch.bmm()是tensor中的一个相乘操作，类似于矩阵中的A*B
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))    # e的形状：（批量大小，序列A的词元数，序列B的词元数）

        beta = torch.bmm(F.softmax(e, dim=-1), B)    # beta的形状：（批量大小，序列A的词元数，embed_size）,意味着序列B被软对齐到序列A的每个词元(beta的第1个维度)

        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)    # alpha的形状：（批量大小，序列B的词元数，embed_size） 序列A被软对齐到序列B的每个词元(alpha的第1个维度)

        return beta, alpha


class Compare(nn.Module):

    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__()
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))    # vA,i是指，所有假设中的词元与前提中词元i软对齐，再与词元i的比较；
        V_B = self.g(torch.cat([B, alpha], dim=2))    # 而vB,j是指，所有前提中的词元与假设中词元i软对齐，再与词元i的比较

        return V_A, V_B


class Aggregate(nn.Module):

    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__()
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):

        V_A = V_A.sum(dim=1)    # 对两组比较向量分别求和
        V_B = V_B.sum(dim=1)

        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))    # 将两个求和结果的连结送到多层感知机中

        return Y_hat


class DecomposableAttention(nn.Module):

    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100, num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__()

        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)

        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)    # 有3种可能的输出：蕴涵、矛盾和中性

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)

        beta, alpha = self.attend(A, B)    # 先对齐
        V_A, V_B = self.compare(A, B, beta, alpha)    # 然后在比较
        y_hat = self.aggregate(V_A, V_B)

        return y_hat


batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = load_data_snli(batch_size, num_steps)

# 预训练好的100维GloVe嵌入来表示输入词元。我们将向量ai和bj在 (15.5.1)中的维数预定义为100。 函数f和中的函数g的输出维度被设置为200.
embed_size, num_hiddens, devices = 100, 200, [torch.device('cuda:0')]
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)


lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)





