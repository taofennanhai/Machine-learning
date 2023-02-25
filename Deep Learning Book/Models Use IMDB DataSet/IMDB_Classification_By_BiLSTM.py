import torch
from torch import nn
from d2l import torch as d2l
from Read_IMDB_DataSet import load_imdb_data

batch_size = 64
train_iter, test_iter, vocab = load_imdb_data(batch_size)    # 数据导入


class BiRNN(nn.Module):                                                                   # **kwargs：发送一个键值对的可变数量的参数列表给函数
    def __init__(self, vocab_size, embedding_size, num_hiddens, num_layers, **kwargs):    # *args：发送一个非键值对的可变数量的参数列表给函数 **kwargs：发送一个键值对的可变数量的参数列表给函数
        super(BiRNN, self).__init__(**kwargs)                                             #    如函数test_all('name','age',name='zxf',age=23)
        self.embedding = nn.Embedding(vocab_size, embedding_size)                         #     ('name', 'age')
                                                                                          #     {'name': 'zxf', 'age': 23}
        self.encoder = nn.LSTM(embedding_size, num_hiddens, num_layers=num_layers, bidirectional=True)    # 设置双向的LSTM
        self.decoder = nn.Linear(4 * num_hiddens, 2)    #

    def forward(self, input):                  # 输入维度是（批量大小，时间步数）
        embedding = self.embedding(input.T)    # 因为长短期记忆网络要求其输入的第一个维度是时间维，所以在获得词元表示之前，输入会被转置。输出形状为（时间步数，批量大小，词向量维度）
        self.encoder.flatten_parameters()      # 参数在内存放置的优化

        outputs, _ = self.encoder(embedding)    # 返回上一个隐藏层在不同时间步的隐状态，outputs的形状是（时间步数，批量大小，2*隐藏单元数）
        out1, out2 = outputs[0], outputs[-1]    # 一个输出是两个LSTM的合并起来的，即(Batch_size,2*hidden_dim)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)     # 连结初始和最终时间步的隐状态，作为全连接层的输入，其形状为（批量大小，4*隐藏单元数）
        outs = self.decoder(encoding)

        return outs


embed_size, num_hiddens, num_layers = 100, 100, 2
devices = d2l.try_all_gpus()
biLSTM = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)


def init_weights(m):    # 初始化网络权重
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)    # 权重初始化为正太分布
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


biLSTM.apply(init_weights)    # 网络权重初始化
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')    # 获取词向量

embeds = glove_embedding[vocab.idx_to_token]
embeds.shape    # torch.Size([49346, 100])

biLSTM.embedding.weight.data.copy_(embeds)    # 使用这些预训练的词向量来表示评论中的词元，并且在训练期间不要更新这些向量。
biLSTM.embedding.weight.requires_grad = False

lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(biLSTM.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(biLSTM, train_iter, test_iter, loss, trainer, num_epochs, devices)