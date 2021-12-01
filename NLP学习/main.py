import torch
import numpy as np
import math

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # x = torch.Tensor([1])
    # y = torch.Tensor([1, 2])
    # x = torch.cat((x, y), dim=0)
    # print(x)

    # rnn = torch.nn.LSTM(10, 20, 2)  # (input_size,hidden_size,num_layers)
    # input = torch.randn(5, 3, 10)    # (seq_len, batch, input_size)
    # h0 = torch.zeros(2, 3, 20)    # (num_layers,batch,output_size)
    # c0 = torch.zeros(2, 3, 20)    # (num_layers,batch,output_size)
    #
    # output, (hn, cn) = rnn(input, (h0, c0))
    #
    # print(output.shape)

    position = torch.arange(0, 10).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, 512, 2).float() * (-math.log(10000.0) / 512))
    pe = torch.zeros(10, 512)  # 初始化位置向量
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    test = [[[1, 2, 3, 4, 17, 18], [5, 6, 7, 8, 19, 20], [9, 10, 11, 12, 21, 22], [13, 14, 15, 16, 23, 24]]]
    test = torch.Tensor(test)
    print(test[:, 0::2])
    print(test[:, 1::2])
    print(test[:2, :])
    print(test.size())
    print(test.size(-1))
    print(position)
