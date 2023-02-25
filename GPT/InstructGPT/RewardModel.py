import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from torchtyping import TensorType


class RewardModel(nn.Module):

    def __init__(self, model_path, dropout):
        super(RewardModel, self).__init__()

        rm = GPT2LMHeadModel.from_pretrained(model_path)

        config = self.model.config
        n_embed = config.n_embd

        # custom head
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_embed, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask)-> TensorType["batch_size", 1]:    # A reward scalar for each item in a batch

        last_hidden_state = self.model(input_ids, attention_mask,).last_hidden_state

        output = self.reward_head(last_hidden_state)

        # for eacb item in the batch
        # choose the hidden state of the last token as a reward!
        reward_scalar = output[:, -1, 0]

        return reward_scalar

