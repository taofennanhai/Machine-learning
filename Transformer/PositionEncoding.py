import math
import torch
import torch.nn as nn
import torch.utils.data as Data


class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 等价于论文写/10000^2i/d_model
        pe[:, 0::2] = torch.sin(position * div_term)  # 每个偶数位置取sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 每个奇数位置取sin

        pe = pe.unsqueeze(0).transpose(0, 1)    # 维度不够下，需要进行扩展
        self.register_buffer('pe', pe)    # pe数组进行反向传播不改变
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]    # 按照第二维相加
        return self.dropout(x)