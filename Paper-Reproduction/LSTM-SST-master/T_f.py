import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy
import torchsummary
from torch.nn import LSTM


class Modle(torch.nn.Module):
    def __init__(self,label_size,layer_num,target_size,dropout):
        super(Modle, self).__init__()
        weight = []
        print("reading glove")
        with open("./data/SST-cla/glove.txt") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                if line=='\n':
                    continue
                data = line.strip().split(' ')[1:]
                if data == []:
                    continue
                sen = [float(i) for i in data]
                weight.append(sen)
                assert len(sen)==300
        del lines
        weights = numpy.array(weight)
        embedding_size = 300
        vocab_size = weights.shape[0]
        # print("vocab_size", vocab_size)
        self.model = LSTM(embedding_size,target_size, layer_num,dropout=dropout,bidirectional=True)
        self.tan=torch.tanh
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.embedding.from_pretrained(torch.from_numpy(weights))
        self.linear1=torch.nn.Linear(2*target_size,target_size)
        self.linear2=torch.nn.Linear(target_size,label_size)
        # print("target_size",target_size)
        self.layer_num=layer_num
        # self.linear2=torch.nn.Linear(seq_len,label_size)

    def forward(self, inputs):
        batch_size=inputs.size(0)
        inputs = self.embedding(inputs)
        # print(inputs.shape)
        inputs=inputs.permute((1,0,2))
        # inputs = inputs.cuda()
        # print("inputs:{}".format(inputs.requires_grad))
        outs,(hidden,cell_status) = self.model(inputs)
        # hidden=hidden.view(self.layer_num, 2, batch_size, -1)
        # hidden=outs[-1]
        outs=outs.permute((1,2,0))
        # cell_status=cell_status.permute((1,0,2))
        # hidden=hidden.reshape(batch_size,-1)
        # print("hidden shape",hidden.shape)
        # cell_status=cell_status.reshape(batch_size,-1)
        # outs=torch.cat((hidden,cell_status),-1)
        # print(outs.shape)
        outs=torch.nn.functional.max_pool1d(outs,outs.size(2)).squeeze(2)
        # hidden1=outs[0]
        # hidden=outs[-1]
        # outs=torch.cat((hidden,hidden1),-1)
        # hidden=hidden.permute((1,0,2))
        # print(out.shape)
        outs=self.tan(outs)
        outs=self.linear1(outs)
        outs=self.linear2(outs)

        # print(l.shape)
        # outs=self.linear2(outs.view(batch_size,-1))
        return outs

