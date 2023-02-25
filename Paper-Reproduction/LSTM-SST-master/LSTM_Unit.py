import torch
from torch.autograd import Variable


class LSTM_Unit(torch.nn.Module):
    def __init__(self, embedding_size):
        super(LSTM_Unit, self).__init__()
        self.weightf = torch.nn.Linear(2 * embedding_size, embedding_size)  # forget gate
        self.weighti = torch.nn.Linear(2 * embedding_size, embedding_size)  # memory gate
        self.weightc = torch.nn.Linear(2 * embedding_size, embedding_size)  # cell status
        self.weighto = torch.nn.Linear(2 * embedding_size, embedding_size)  # out status
        self.sigmoid = torch.nn.Sigmoid()
        self.cell_status=Variable(torch.zeros((1, embedding_size)), requires_grad=True).cuda()
        self.tanh = torch.nn.Tanh()

    def forward(self, state: torch.Tensor, input: torch.Tensor, cons: torch.Tensor):
        """

        :param state: last state
        :param input: word embedding
        :param cons: reference
        """
        # print("state",state.shape)
        # print("input",input.shape)

        concat_status = torch.cat((state, input), -1)
        forget_status = self.weightf(concat_status)
        forget_status = self.sigmoid(forget_status)
        mem_gate = self.sigmoid(self.weighti(concat_status))
        tmp_cell_status = self.tanh(self.weightc(concat_status))
        cell_gate = mem_gate.mul(tmp_cell_status)
        out_cell_status = cons.mul(forget_status) + cell_gate
        out_hidden = self.tanh(out_cell_status).mul(forget_status)
        return out_hidden, out_cell_status
