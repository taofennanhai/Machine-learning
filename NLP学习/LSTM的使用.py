import torch
import numpy as np

batch_size = 3
hidden_size = 5
embedding_dim = 6
seq_length = 4  # 序列长度可以认为是LSTM单元循环长度
num_layers = 2
num_directions = 1
vocab_size = 20

# input_size：x的特征维度hidden_size：隐藏层的特征维度num_layers：lstm隐层的层数，默认为1bias：False则bih = 0
# 和bhh = 0.
# 默认为False batch_first：True则输入输出的数据格式为(batch, seq, feature)
# dropout：除最后一层，每一层的输出都进行dropout，默认为: 0
# bidirectional：True则为双向lstm默认为False输入：input, (h0, c0)
# 输出：output, (hn, cn)

lstm_layer = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, bias=True,
                           batch_first=True, bidirectional=False)

# input(seq_len, batch, input_size)
# h0(num_layers * num_directions, batch, hidden_size)
# c0(num_layers * num_directions, batch, hidden_size)

input_data = np.random.uniform(0, 19, size=(batch_size, seq_length))
input_data = torch.from_numpy(input_data).long()

embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
input_data = embedding_layer(input_data)

# output(seq_len, batch, hidden_size * num_directions)
# hn(num_layers * num_directions, batch, hidden_size)
# cn(num_layers * num_directions, batch, hidden_size)

output, (h_n, c_n) = lstm_layer(input_data)
print(output)
print(output.size())  # 输出的是3X4X5矩阵,每个4X5矩阵意味着一个样例,4是LSTM循环长度即是每个LSTM的输出，5是输出维度

h_n = h_n.squeeze(0)  # 其中squeeze(0)代表若第一维度值为1则去除第一维度，squeeze(1)代表若第二维度值为1则去除第二维度。去除size为1的维度，
                      # 包括行和列。当维度大于等于2时，squeeze()无作用。
print(h_n)
print(h_n.size())     # 大小为1X3X5, 1指的是层数乘上方向,3是batch,5就是输出维度
