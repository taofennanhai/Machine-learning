import torch
import torch.nn as nn
import numpy as np


class FeedForwardNet(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, device='cuda'):    # Pytorch中的Linear只会对最后一维操作，所以正好是我们希望的每个位置用同一个全连接网络
        super(FeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device

        self.fc = nn.Sequential(    # 序列组合进行传播
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.device).forward(output + residual)  # [batch_size, seq_len, d_model] 进行归一化和残差链接