import math
import torch
import torch.nn as nn

# Transformer Parameters
word_embedding_dim = 512  # Embedding Size（token embedding和position编码的维度）
linear_projection = 2048  # FeedForward dimension (两次线性层中的隐藏层 512->2048->512，线性层是用来做特征提取的），当然最后会再接一个projection层
Q_dim = K_dim = V_dim = 64  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
block_layers = 6  # number of Encoder of Decoder Layer（Block的个数）
multi_heads = 8  # number of heads in Multi-Head Attention（有几套头）




class PositionalEncoding(nn.Module):    # 位置编码
    def __init__(self, word_embedding_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)    # 将元素置为0的概率为p

        pe = torch.zeros(max_len, word_embedding_dim)    # 初始化位置向量
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    # 设置每个位置为(max_len,1)
        div_term = torch.exp(torch.arange(0, word_embedding_dim, 2).float() * (-math.log(10000.0) / word_embedding_dim))  # 等价于1/10000^(2i/d_model)

        pe[:, 0::2] = torch.sin(position * div_term)    # 取pe矩阵的偶数列
        pe[:, 1::2] = torch.cos(position * div_term)    # 取pe矩阵的奇数列
        pe.unsqueeze(0).transpose(0, 1)    # 输入的每个句子向量是 [seq_len,batch,word_embedding]
        self.register_buffer('pe', pe)     # buffer一种是反向传播不需要被optimizer更新

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]    # x的维度为[seq_len,batch,word_embedding]
        return self.dropout(x)    # 进行随机失活


def get_attention_pad_mask(seq_q, seq_k):
    # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
    """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
    encoder和decoder都可能调用这个函数，所以seq_len视情况而定
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be target_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """

    batch_size, len_q = seq_q.size()    # 获取每个源句子的尺寸
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    pad_attention_mask = seq_k.data.eq(0).unsqueeze(1)    # [batch_size, 1, len_k], True is masked
    return pad_attention_mask.expand(batch_size, len_q, len_k)    # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)


def get_attetion_subsequence_mask(seq):
    """建议打印出来看看是什么的输出（一目了然）
        seq: [batch_size, tgt_len]
    """

