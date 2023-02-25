import torch
from torch import nn
from d2l import torch as d2l
from Read_IMDB_DataSet import load_imdb_data

batch_size = 64
train_iter, test_iter, vocab = load_imdb_data(batch_size)

X, K = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])


def corr1d(X, K):    # 定义文本的一维卷积
    w = K.shape[0]
    Y = torch.zeros(X.shape[0]-w+1)    # 一维卷积后的长度
    for i in range(Y.shape[0]):
        Y[i] = (X[i:i+w]*K).sum()      # 两个维度相乘后在相加
    return Y


def multi_channel_corr1d(X, K):    # 定义文本多通道的一维卷积
    channel_num = X.shape[0]  # 先获取第一个通道维度的数量
    Y = torch.zeros(channel_num, X.shape[1]-K.shape[1]+1)    # 定义卷积后多通道的文本特征

    i = 0
    for x, k in zip(X, K):    # 先遍历每个通道
        Y[i, ] = corr1d(x, k)    # 给每个通道赋值
        i = i+1
    return Y.sum(dim=0)    # 把每个通道的维度加成一个维度


X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = torch.tensor([[1, 2], [3, 4], [-1, -3]])
test = multi_channel_corr1d(X, K)
print()


class TextCNN(nn.Module):    # 定义文本CNN模型
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)

        self.constant_embedding = nn.Embedding(vocab_size, embed_size)    # 这个嵌入层不需要训练
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)

        # AdaptiveAvgPool1d(N) 对于一个输入的tensor的最后一维进行pool，如[2,3,5]->[2,3,1]
        self.pool = nn.AdaptiveAvgPool1d(1)    # 最大时间汇聚层没有参数，因此可以共享此实例
        self.relu = nn.ReLU()

        self.convs = nn.ModuleList()    # 创建多个一维卷积层

        # 为什么是两倍？因为嵌入层有两个    in_channels(int) – 输入信号的通道。在文本分类中，即为词向量的维度
        # out_channels(int) – 卷积产生的通道。有多少个out_channels，就需要多少个1维卷积
        # kernel_size(int or tuple) - 卷积核的尺寸，卷积核的大小为(k,)，第二个维度是由in_channels来决定的，所以实际上卷积大小为kernel_size*in_channels
        # 再输入一维卷积的时候，需要将32*35*256变换为32*256*35，因为一维卷积是在最后维度上扫的，最后out的大小即为：32*100*（35-2+1）=32*100*34
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2*embed_size, c, k))

    def forward(self, input):
        # 沿着向量维度将两个嵌入层连结起来，
        # 每个嵌入层的输出形状都是（批量大小，词元数量，词元向量维度）连结起来
        embeddings = torch.cat((self.constant_embedding(input), self.embedding(input)), dim=2)

        embeddings = embeddings.permute(0, 2, 1)    # 根据一维卷积层的输入格式，重新排列张量，以便通道作为第2维 和图中一样

        # 分别经过两个卷积和的特征提取
        # 每个一维卷积层在最大时间汇聚层合并后，获得的张量形状是（批量大小，通道数，1） 删除最后一个维度并沿通道维度连结
        encoding = torch.cat([torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1) for conv in self.convs], dim=1)

        outputs = self.decoder(self.dropout(encoding))
        return outputs


embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)


def init_weights(m):    # 初始化Linear和卷积核的权重
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)


glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]

net.embedding.weight.data.copy_(embeds)    # 一份可嵌入参数是可学习的
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False    # 第二份参数固定


lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)    # 模型训练

print("train complete")
d2l.predict_sentiment(net, vocab, 'this movie is so great')