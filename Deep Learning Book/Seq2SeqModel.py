import collections
import math
import torch
from torch import nn
from d2l import torch as d2l


input_dim = 100
output_dim = 200


test = torch.zeros(64, 100, 100)


decoder = nn.GRU(input_size=input_dim, hidden_size=output_dim, batch_first=True, num_layers=2)

# h: shape = [num_layers * num_directions, batch, hidden_size]的张量 h包含的是句子的最后一个单词的隐藏状态
# c: 与h的形状相同，它包含的是在当前这个batch_size中的每个句子的初始细胞状态。h,c如果不提供，那么默认是0
test, h = decoder(test)

print(test)


class Seq2Seq_Encoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0, **kwargs):
        super(Seq2Seq_Encoder, self).__init__(**kwargs)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim,
                               num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, encoder_input, *args):    # 这里的arg会接收每个句子的有效长度
        # input的形状：(batch_size,num_steps,embed_size)
        encoder_input = self.embedding(encoder_input)

        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        output, h = self.encoder(encoder_input)    # output的形状:(num_steps,batch_size,num_hiddens)

        return output, h


class Seq2Seq_Decoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0, **kwargs):
        super(Seq2Seq_Decoder, self).__init__(**kwargs)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder = nn.GRU(input_size=embed_dim + hidden_dim, hidden_size=hidden_dim, batch_first=True,
                               num_layers=num_layers, dropout=dropout)    # 这里的输入是把embedding和hidden做一个拼接
        self.dense = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, decoder_input, h):    # h，c是encoder的输出
        # decoder_input的形状：(batch_size,num_steps,embed_size) h的形状:(num_layers,batch_size,num_hiddens)
        decoder_input = self.embedding(decoder_input)

        # repeat就是按位置乘以倍数 先取最后一个的hidden_embedding，然后在把变成[batch_size, seq_len, hidden]形状
        context = h[-1].unsqueeze(1).repeat(1, decoder_input.shape[1], 1)    # 这里应该用h作为上下文的环境就是context
        decoder_input_and_context = torch.cat((decoder_input, context), dim=2)    # 拼接输入和encoder

        output, h = self.decoder(decoder_input_and_context, h)
        output = self.dense(output)

        # output的形状:(batch_size,num_steps,vocab_size)
        # state形状:(num_layers,batch_size,num_hiddens)
        return output, h


# encoder = Seq2Seq_Encoder(vocab_size=10, embed_dim=8, hidden_dim=16, num_layers=2)
# encoder.eval()
# X = torch.zeros((4, 7), dtype=torch.long)
# output, h = encoder(X)
# output.shape
#
#
# decoder = Seq2Seq_Decoder(vocab_size=10, embed_dim=8, hidden_dim=16, num_layers=2)
# decoder.eval()
# state = decoder.init_state(encoder(X))
# test1 = h[-1]
# output, h = decoder(X, h)
# output.shape, state.shape


def sequence_mask(X, valid_len, value=0):

    maxlen = X.size(1)    # 定义最大长度
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]

    X[~mask] = value
    return X


# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# test2.py = sequence_mask(X, torch.tensor([1, 2]))
#
#
# X = torch.ones(2, 3, 4)
# test3 = sequence_mask(X, torch.tensor([1, 2]), value=-1)


class MaskedSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):    # """带遮蔽的softmax交叉熵损失函数"""

    # pred的形状：(batch_size,num_steps,vocab_size) label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)

        self.reduction = 'none'

        # pred.permute(0, 2, 1)的形状(batch_size ,vocab_size, num_steps)
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)    # 就是调用交叉熵的函数
        weighted_loss = (unweighted_loss * weights).mean(dim=1)    # 取对应部分的值求和除以num_steps

        return weighted_loss


loss = MaskedSoftmaxCrossEntropyLoss()
test = loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([2, 1, 0]))

print(loss)


#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):    # 初始化参数
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:    # rnn的参数必须逐个初始化
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCrossEntropyLoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])

    for epoch in range(num_epochs):    # 训练代数
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()

            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)   # decoder每句话都加上BOS

            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学 把bos加在句子的前头

            Y_hat, _ = net(X, dec_input, X_valid_len)    # Y_hat输出每个的词的概率

            l = loss(Y_hat, Y, Y_valid_len)    # 用交叉熵计算有效长度的每个词的概率

            l.sum().backward()      # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} ' f'tokens/sec on {str(device)}')


embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()


train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2Seq_Encoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2Seq_Decoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)    # 训练模型



# 进行模型预测
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]

    enc_valid_len = torch.tensor([len(src_tokens)], device=device)    # 进行attention的时候需要
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])

    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)    # 输入encoder

    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)    # 获取encoder最后一个状态
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)

    output_seq, attention_weight_seq = [], []

    for _ in range(num_steps):    # 循环时间步数的decoder

        Y, dec_state = net.decoder(dec_X, dec_state)

        # 使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)

        pred = dec_X.squeeze(dim=0).type(torch.int32).item()

        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)

        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)

    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .', 'i like dog .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .', 'j\'aime les chiens .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)

    print(f'{eng} => {translation}')
