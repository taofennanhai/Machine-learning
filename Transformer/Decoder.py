import torch
import torch.nn as nn
import numpy as np

from Encoder import get_attention_pad_mask, get_attention_subsequence_mask, MultiHeadAttention
from FeedForwardNet import FeedForwardNet
from PositionEncoding import PositionEncoding


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.decoder_self_attention = MultiHeadAttention()    # 第一层的掩码多头注意力
        self.decoder_encoder_attention = MultiHeadAttention()    # 第二层encoder输入和decoder的掩码的多头注意力
        self.pos_ffn = FeedForwardNet()    # 归一化加残差链接层

    def forward(self, decoder_inputs, encoder_outputs, decoder_self_attention_mask, decoder_encoder_attention_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """

        # decoder_outputs: [batch_size, tgt_len, d_model], decoder_self_attention: [batch_size, n_heads, tgt_len, tgt_len]    第一层输入的是自身的掩码加上上三角掩码矩阵
        decoder_outputs, decoder_self_attention = self.decoder_self_attention(decoder_inputs, decoder_inputs, decoder_inputs, decoder_self_attention_mask)  # 这里的Q,K,V全是Decoder自己的输入

        # decoder_outputs: [batch_size, tgt_len, d_model], decoder_encoder_attention: [batch_size, h_heads, tgt_len, src_len]   第二层输入的是encoder和decoder的掩码
        decoder_outputs, decoder_encoder_attention = self.decoder_encoder_attention(decoder_outputs, encoder_outputs, encoder_outputs, decoder_encoder_attention_mask)  # Attention层的Q(来自decoder) 和 K,V(来自encoder)

        decoder_outputs = self.pos_ffn(decoder_outputs)  # [batch_size, tgt_len, d_model] 输出之后经过前向传播加上归一化加残差链接
        return decoder_outputs, decoder_self_attention, decoder_encoder_attention  # decoder_self_attention, dec_enc_attn这两个是为了可视化的



class Decoder(nn.Module):
    def __init__(self, target_vocab_size=9, d_model=512, device='cuda', n_layers=6):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.device = device
        self.n_layers = n_layers

        self.target_emb = nn.Embedding(target_vocab_size, d_model)  # Decoder输入的embed词表
        self.pos_emb = PositionEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])  # Decoder的多块blocks层

    def forward(self, decoder_inputs, encoder_inputs, encoder_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]   # 用在Encoder-Decoder Attention层
        """
        decoder_outputs = self.target_emb(decoder_inputs)  # [batch_size, tgt_len, d_model]  获得词嵌入

        decoder_outputs = self.pos_emb(decoder_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]  获得位置编码

        # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
        decoder_self_attention_pad_mask = get_attention_pad_mask(decoder_inputs, decoder_inputs).to(self.device)  # [batch_size, tgt_len, tgt_len]

        # Masked Self_Attention：当前时刻是看不到未来的信息的  获得上三角矩阵掩码
        decoder_self_attention_subsequence_mask = get_attention_subsequence_mask(decoder_inputs).to(self.device)  # [batch_size, tgt_len, tgt_len]

        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）  gt函数是比较大小的，其中大于0的返回True
        decoder_self_attention_mask = torch.gt((decoder_self_attention_pad_mask + decoder_self_attention_subsequence_mask), 0).to(self.device)  # [batch_size, tgt_len, tgt_len]; torch.gt比较两个矩阵的元素，大于则返回1，否则返回0

        # 这个mask主要用于encoder-decoder attention层 因为decoder的数据不需要看到encoder的padding,attention设置为负无穷
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        decoder_encoder_attention_mask = get_attention_pad_mask(decoder_inputs, encoder_inputs)  # [batch_size, tgt_len, src_len]


        decoder_self_attentions, decoder_encoder_attentions = [], []    # 画每个词之间的热力图

        for layer in self.layers:
            # decoder_outputs: [batch_size, tgt_len, d_model], decoder_self_attention: [batch_size, n_heads, tgt_len, tgt_len], decoder_encoder_attention: [batch_size, h_heads, tgt_len, src_len]
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            decoder_outputs, decoder_self_attention, decoder_encoder_attention = layer(decoder_outputs, encoder_outputs, decoder_self_attention_mask, decoder_encoder_attention_mask)

            decoder_self_attentions.append(decoder_self_attention)
            decoder_encoder_attentions.append(decoder_encoder_attention)

        # decoder_outputs: [batch_size, tgt_len, d_model]
        return decoder_outputs, decoder_self_attentions, decoder_encoder_attentions
