import torch
import torch.functional as F
import torch.utils.data as Data
import torch.nn as nn

# 准备数据集
corpus = ['he is a king',
          'she is a queen',
          'he is a man',
          'she is a woman',
          'warsaw is poland capital',
          'berlin is germany capital',
          'paris is france capital']


word_list = " ".join(corpus).split()    # 将上面的单词逐个分开
word_list = list(set(word_list))    # 将分词后的结果去重
word_dict = {w: i for i, w in enumerate(word_list)}    # 对单词建立索引，for循环里面是先取索引，再取单词
number_dict = {i: w for i, w in enumerate(word_list)}    # 反向建立索引

n_class = len(word_dict)    # 计算词典长度
n_hidden = 2     # 隐藏层神经单元数目,就是H
m = 2           # 词向量维度
n_step = 3      # 代表n-gram模型 2就是2-gram 用前两个词预测最后一个词


def make_batch(corpus):    # 构建输入输出数据
    input_batch = []
    target_batch = []

    for sentance in corpus:

        word = sentance.split()    # 将句子中每个词分词

        input = [word_dict[n] for n in word[:-1]]        # :-1表示取每个句子里面的前两个单词作为输入, 然后通过word_dict取出这两个单词的下标，作为整个网络的输入

        target = word_dict[word[-1]]    # target取的是预测单词的下标，这里就是cat,coffee和milk

        input_batch.append(input)

        target_batch.append(target)

    return input_batch, target_batch


input_batch, target_batch = make_batch(corpus)

input_batch = torch.LongTensor(input_batch)    # 将数据装载到torch上
target_batch = torch.LongTensor(target_batch)


dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset=dataset, batch_size=5, shuffle=True)
dtype = torch.FloatTensor

# 定义网络结构
# nn.Parameter() 的作用是将该参数添加进模型中，使其能够通过 model.parameters() 找到、管理、并且更新
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        # 计算词向量表，大小是len(word_dict) * m

        self.C = nn.Embedding(n_class, m)
        # 下面就是初始化网络参数，公式如下
        """
        hiddenout = tanh(d + X*H)
        y = b + X*H + hiddenout*U
        """
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        self.b = nn.Parameter(torch.randn(n_class).type(dtype))

    def forward(self, X):
        '''
        X: [batch_size, n_step]
        '''
        # 根据词向量表，将我们的输入数据转换成三维数据
        # 将每个单词替换成相应的词向量

        X = self.C(X)  # [batch_size, n_step] => [batch_size, n_step, m]
        # 将替换后的词向量表的相同行进行拼接
        # view的第一个参数为-1表示自动判断需要合并成几行
        X = X.view(-1, n_step * m)  # [batch_size, n_step * m]
        hidden_out = torch.tanh(self.d + torch.mm(X, self.H))  # [batch_size, n_hidden]
        output = self.b + torch.mm(X, self.W) + torch.mm(hidden_out, self.U)  # [batch_size, n_class]
        return output

model = NNLM()
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    for batch_x, batch_y in loader:
        optim.zero_grad()

        # output : [batch_size, n_class], batch_y : [batch_size] (LongTensor, not one-hot)
        output = model.forward(batch_x)

        loss = criterion(output, batch_y)

        # 每1000次打印一次结果
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))


        # 反向传播更新梯度
        loss.backward()
        optim.step()


# Predict
# max()取的是最内层维度中最大的那个数的值和索引，[1]表示取索引
predict = model(input_batch).data.max(1, keepdim=True)[1]

print(predict)