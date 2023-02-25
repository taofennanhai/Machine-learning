from typing import List, Tuple

import torch
import numpy as np
from torch import jit
from torch.autograd import Variable


class LSTM_Unit(jit.ScriptModule):
    def __init__(self,embedding_size,hidden_size):
        super(LSTM_Unit, self).__init__()
        self.linear=torch.nn.Linear(embedding_size,4*hidden_size)
        self.linear_hidden=torch.nn.Linear(hidden_size,4*hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        # self.cell_status=Variable(torch.zeros((1, embedding_size)), requires_grad=True).cuda()
        self.tanh = torch.nn.Tanh()
        self.layernormal1=torch.nn.LayerNorm(4*hidden_size)
        self.layernormal2=torch.nn.LayerNorm(4*hidden_size)
        self.layernormal3=torch.nn.LayerNorm(hidden_size)
        # print("hidden_size")
    @jit.script_method
    def forward(self, state: torch.Tensor, input: torch.Tensor, cons: torch.Tensor):
        """

        :param state: last state
        :param input: word embedding
        :param cons: reference
        """
        # print("state",state.shape)
        # print("input",input.shape)
        # print(input.shape)
        # print(state.shape)
        feature=self.layernormal1(self.linear(input))+self.layernormal2(self.linear_hidden(state))
        # feature=self.linear(input)+self.linear_hidden(state)
        feature.squeeze()
        ingate,forget_gate,cell_gate,cellstat=feature.chunk(4,1)
        ingate=self.sigmoid(ingate)
        forget_gate=self.sigmoid(forget_gate)
        cell_gate=self.tanh(cell_gate)
        cellstat=self.sigmoid(cellstat)
        cell=self.layernormal3(torch.mul(cons,forget_gate)+torch.mul(ingate,cell_gate))
        # cell=torch.mul(cons,forget_gate)+torch.mul(ingate,cell_gate)
        out=torch.mul(self.tanh(cell),cellstat)
        return out,cell


class LSTM_layer(jit.ScriptModule):
    def __init__(self, embeddings_size, embe_size):#, label_size):
        super(LSTM_layer, self).__init__()
        # self.layer_num=layer_num
        self.model=LSTM_Unit(embeddings_size,embe_size)
        # self.predict_linear=torch.nn.Linear(embe_size,label_size)
        self.hidden_size=embe_size
        self.layernormal=torch.nn.LayerNorm
        # self.dropout=torch.nn.Dropout(dropoutrate)
    @jit.script_method
    def forward(self, inputs: torch.Tensor,fini_state,finit_hidden):
        # print(input.shape)
        # batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        # embedding_size = inputs.size(2)
        inputs = inputs.permute([1, 0, 2])
        # next_inputs=inputs
        hidden=finit_hidden
        state=fini_state
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(seq_len):
            # if j!=0:
                # hidden=self.dropout(hidden)
            hidden,state=self.model(hidden,inputs[i],state)
            # hidden=self.dropout(hidden)
            outputs+=[hidden]
        output=torch.stack(outputs,0)
        output=output.permute([1,0,2])#batch_size,seq_len,embedding_size
        # print("output_state4:{}".format(output.requires_grad))
        return output,state
class reverse_LSTM(jit.ScriptModule):
    def __init__(self, embeddings_size,embe_size):
        super(reverse_LSTM, self).__init__()
        # self.layer_num=layer_num
        self.model=LSTM_Unit(embeddings_size,embe_size)
        # self.predict_linear=torch.nn.Linear(embe_size,label_size)
        # self.hidden_size=embe_size
    def reverse(self,inputs:List[torch.Tensor]):
        inputs.reverse()
        return inputs
    @jit.script_method
    def forward(self, inputs: torch.Tensor,fini_state,finit_hidden):
        # print(input.shape)
        # batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        # embedding_size = inputs.size(2)
        inputs = inputs.permute([1, 0, 2])
        inputs=inputs.unbind(0)
        # print("inputs",type(inputs))
        input=self.reverse(inputs)
        # next_inputs=inputs
        hidden=finit_hidden
        state=fini_state
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(seq_len):
            # if j!=0:
                # hidden=self.dropout(hidden)
            hidden,state=self.model(hidden,input[i],state)
            # hidden=self.dropout(hidden)
            outputs+=[hidden]
        output=torch.stack(self.reverse(outputs))
        output=output.permute([1,0,2])#batch_size,seq_len,embedding_size
        # print("output_state4:{}".format(output.requires_grad))
        return output,state
class Stack_LSTM(jit.ScriptModule):
    def __init__(self, layer_num, embedding_size, hidden_size,Dropout_rate):
        super(Stack_LSTM,self).__init__()
        self.layer_num=layer_num
        self.dropoutf = torch.nn.ModuleList([torch.nn.Dropout(Dropout_rate) for i in range(self.layer_num)])
        self.dropoutb = torch.nn.ModuleList([torch.nn.Dropout(Dropout_rate) for i in range(self.layer_num)])
        if layer_num>1:
            self.forward_model=torch.nn.ModuleList([LSTM_layer(embedding_size,hidden_size)]+[LSTM_layer(hidden_size,hidden_size) for i in range(self.layer_num-1)])
            self.backward_model=torch.nn.ModuleList([reverse_LSTM(embedding_size,hidden_size)]+[reverse_LSTM(hidden_size,hidden_size) for i in range(self.layer_num-1)])
        else:
            self.forward_model = torch.nn.ModuleList(
                [LSTM_layer(embedding_size, hidden_size)])
            self.backward_model = torch.nn.ModuleList(
                [reverse_LSTM(embedding_size, hidden_size)])
        self.dropout=torch.nn.Dropout(Dropout_rate)
    @jit.script_method
    def forward(self, inputs,fhiddens, bhiddens, fstates, bstates):
        outputs=jit.annotate(List[Tuple[torch.Tensor,torch.Tensor]],[])
        outputf=inputs
        outputb=inputs
        i=0
        for layer1,layer2,dropout1,dropout2 in zip(self.forward_model,self.backward_model,self.dropoutf,self.dropoutb):
            fstate=fstates[i]
            bstate=bstates[i]
            fhidden=fhiddens[i]
            bhidden=bhiddens[i]
            outputf,fstate=layer1(outputf,fstate,fhidden)
            outputb,bstate=layer2(outputb,bstate,bhidden)
            outputf=dropout1(outputf)
            outputb=dropout2(outputb)
            i+=1
            outputs+=[(fstate,bstate)]
        return torch.cat((outputf,outputb),-1),outputs

