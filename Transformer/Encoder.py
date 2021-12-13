import torch
import torch.nn as nn
import numpy as np

from FeedForwardNet import FeedForwardNet
from PositionEncoding import PositionEncoding


def get_attention_pad_mask(seq_q, seq_k):
    # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
    """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
    encoder和decoder都可能调用这个函数，所以seq_len视情况而定
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """

    batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    pad_attention_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked, eq函数判断是否为0，是就返回true

    # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)
    return pad_attention_mask.expand(batch_size, len_q, len_k)  # 扩展成和输入句子相同的mask矩阵


def get_attention_subsequence_mask(seq):
    """建议打印出来看看是什么的输出（一目了然）
    seq: [batch_size, tgt_len]
    """
    attention_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attention_shape: [batch_size, tgt_len, tgt_len]

    subsequence_mask = np.triu(np.ones(attention_shape), k=1)  # 生成一个上三角矩阵, k=1对角线向上移动一个格子，k=-1为向下移动，K=0不变
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()    # byte函数是非0即1

    return subsequence_mask  # [batch_size, tgt_len, tgt_len]



class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k=64):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = 64

    def forward(self, Q, K, V, attention_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
                0          1         2           3
               -4          -3        -2          -1
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)   # QK矩阵相乘

        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        scores.masked_fill_(attention_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attention = nn.Softmax(dim=-1).forward(scores)  # 对最后一个维度(v)做softmax,也就是一个矩阵的一行

        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        context = torch.matmul(attention, V)  # context: [batch_size, n_heads, len_q, d_v]

        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        return context, attention


class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    """

    def __init__(self, d_model=512, d_v=64, d_k=64, n_heads=8, device='cuda'):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.d_v = d_v
        self.d_k = d_k
        self.n_heads = n_heads
        self.device = device

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)  # 这里是多头注意力机制的矩阵，不过拼在一起了,每个头的维度大小为d_model/n_heads
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)   # 把多头计算完的注意力映射回d_model维度


    def forward(self, input_Q, input_K, input_V, attention_mask):    # 输入三次，是为了方便在encoder和decoder复用，decoder那块需要encoder的输入
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """

        residual, batch_size = input_Q, input_Q.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头

        # Q: [batch_size, n_heads, len_q, d_k]    其中QKV都是X改变过来的
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 因为是多头，所以mask矩阵要扩充成4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attention: [batch_size, n_heads, len_q, len_k]
        context, attention = ScaledDotProductAttention().forward(Q, K, V, attention_mask)    # 进行QKV矩阵相乘

        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)    # 每个头的注意力拼接起来

        # 再做一个projection
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(self.device).forward(output + residual), attention


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.encoder_self_attention = MultiHeadAttention()
        self.pos_ffn = FeedForwardNet()

    def forward(self, encoder_inputs, encoder_self_attention_mask):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # encoder_outputs: [batch_size, src_len, d_model], attention: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        encoder_outputs, attention = self.encoder_self_attention(encoder_inputs, encoder_inputs, encoder_inputs, encoder_self_attention_mask)  # enc_inputs to same Q,K,V（未线性变换前）
        encoder_outputs = self.pos_ffn(encoder_outputs)
        # encoder_outputs: [batch_size, src_len, d_model]
        return encoder_outputs, attention


class Encoder(nn.Module):
    def __init__(self, src_vocab_size=6, d_model=512, n_layers=6):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  # token Embedding
        self.pos_emb = PositionEncoding(d_model)  # Transformer中位置编码时固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])


    def forward(self, encoder_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        encoder_outputs = self.src_emb(encoder_inputs)  # [batch_size, src_len, d_model]
        encoder_outputs = self.pos_emb(encoder_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]

        # Encoder输入序列的pad mask矩阵, 不需要关注的pad的符号
        encoder_self_attention_mask = get_attention_pad_mask(encoder_inputs, encoder_inputs)  # [batch_size, src_len, src_len]
        encoder_self_attentions = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系

        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # encoder_outputs: [batch_size, src_len, d_model], encoder_self_attention: [batch_size, n_heads, src_len, src_len]
            encoder_outputs, encoder_self_attention = layer(encoder_outputs, encoder_self_attention_mask)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention

            encoder_self_attentions.append(encoder_self_attention)  # 这个只是为了可视化

        return encoder_outputs, encoder_self_attentions