import math

import torch
import torch.nn as nn
from EncoderLayer import EncoderLayer, get_attention_pad_mask


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))



class Embedding(nn.Module):

    def __init__(self, vocab_size=40, d_model=512, max_len=30, n_segments=2):
        super(Embedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)    # 单词嵌入
        self.position_embedding = nn.Embedding(max_len, d_model)    # 位置嵌入
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        sequence_len = x.size(1)    # 获得句子长度
        pos = torch.arange(sequence_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len] 扩展位置向量维度

        embedding = self.token_embedding(x) + self.position_embedding(pos) + self.seg_embed(seg)    # 三个向量相加
        return self.layernorm(embedding)    # 对每个样本进行layernorm



class Bert(nn.Module):
    def __init__(self, n_layers=6, d_model=512, vocab_size=40):
        super(Bert, self).__init__()
        self.d_model = d_model
        self.embedding = Embedding()    # 词嵌入加上位置嵌入加上段落嵌入
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])    # 多层encoder

        self.fc = nn.Sequential(    # cls输出层进行一个线性变换
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh()
        )
        self.classifier = nn.Linear(d_model, 2)    # cls线性变换之后的映射成类别种类问题，这里就是是否为下一个句子

        self.linear = nn.Linear(d_model, d_model)    # MLM语言模型也需要进行线性变换
        self.activ2 = gelu    # 不是使用relu函数而是gelu函数

        # fc2 is shared with embedding layer
        embed_weight = self.embedding.token_embedding.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)    # 预测单词
        # self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):    # 输入组合句子，输入段落编号，在输入mask单词的位置[0,0,0, 1,1,1]

        # [bach_size, seq_len, d_model]
        output = self.embedding(input_ids, segment_ids)    # bert的第一步操作，输入的是单词，然后加上位置编码，加上段落编码

        # [batch_size, maxlen, maxlen]
        encoder_self_attention_mask = get_attention_pad_mask(input_ids, input_ids)    # 获得input两个句子后面的掩码

        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, encoder_self_attention_mask)    # 进行一层的encoder


        # [batch_size, d_model] it will be decided by first token(CLS)
        h_pooled = self.fc(output[:, 0])    # 拿第一个cls的向量进行线性变换
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] 通过上面线性变换的cls向量predict isNext

        # [batch_size, max_pred, d_model]
        masked_pos = masked_pos[:, :, None].expand(-1, -1, self.d_model)    # 把mask_pos扩展至三维,计算交叉熵loss时候方便
        h_masked = torch.gather(output, 1, masked_pos)    # 按照mask单词顺序的位置，把output的顺序进行调换，方便计算交叉熵

        # [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked))     # 先进行线性变换，然后进行gelu变换
        logits_lm = self.fc2(h_masked)    # [batch_size, max_pred, vocab_size] 在进行一次线性变换

        return logits_lm, logits_clsf    # 第一个是每个mask单词的向量，用来预测单词，第二个是判断是否为相邻句子

