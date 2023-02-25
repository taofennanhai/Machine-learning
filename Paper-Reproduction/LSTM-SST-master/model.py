import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy
import torchsummary
from Bi_LSTM import Stack_LSTM


class Modle(torch.nn.Module):
    def __init__(self, seq_len, label_size,layer_num,target_size,dropout_rate):
        super(Modle, self).__init__()
        weight = []
        print("reading glove")
        with open("./data/SST-cla/glove.txt") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                if line=='\n':
                    continue
                data = line.strip().split(' ')[1:]
                sen = [float(i) for i in data]
                weight.append(sen)
                assert len(sen)==300
        del lines
        self.layer_num=layer_num
        self.target_size=target_size
        self.convert=torch.nn.Linear(300,target_size)
        # self.dropout=torch.nn.Dropout(dropout_rate)
        weights = numpy.array(weight)
        vocab_size = weights.shape[0]
        print("vocab_size", vocab_size)
        self.model = Stack_LSTM(layer_num,300,target_size,dropout_rate)
        self.predict=torch.nn.Linear(2*target_size,label_size)
        # self.embedding = torch.nn.Embedding(vocab_size, 300)
        # self.embedding.weight.requires_grad
        self.embedding=torch.nn.Embedding.from_pretrained(torch.from_numpy(weights))
        # self.LSTM = LSTM(target_size,layer_num,seq_len,label_size,dropout_rate)
        self.sig=torch.nn.Sigmoid()
        # self.pool=torch.nn.MaxPool1d()
    # @profile
    def forward(self, inputs):
        fbegin_state= Variable(torch.zeros(self.layer_num,inputs.size(0), self.target_size).cuda())
        fhidden_state=Variable(torch.zeros(self.layer_num,inputs.size(0), self.target_size).cuda())
        bbegin_state= Variable(torch.zeros(self.layer_num,inputs.size(0), self.target_size).cuda())
        bhidden_state=Variable(torch.zeros(self.layer_num,inputs.size(0), self.target_size).cuda())
        inputs = self.embedding(inputs)
        inputs = inputs.cuda()
        # inputs=self.convert(inputs)
        # print("inputs:{}".format(inputs.requires_grad))
        outs,states = self.model(inputs,fbegin_state,fhidden_state,bbegin_state,bhidden_state)
        outs=outs.permute([0,2,1])
        outs = torch.nn.functional.max_pool1d(outs, outs.size(2)).squeeze(2)
        # print(outs.shape)
        outs=self.predict(outs)
        # outs=self.sig(outs)
        return outs
