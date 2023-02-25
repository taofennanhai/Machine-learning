import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import optimizer

from copy import deepcopy
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2PreTrainedModel, AutoModelForCausalLM
from Read_IMDB_DataSet import read_imdb
from Agent import Agent

model_path = '../../../Model/GPT2-base'


gpt2 = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


dataset = read_imdb('../../dataset/aclImdb', True)

model = Agent(gpt2)
ref_model = deepcopy(gpt2).eval()

max_new_tokens = 20
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.8,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": max_new_tokens
}


config = RLHFConfig()
N_EPOCH = 100
trainer = RLHFTrainer(model, ref_model, config)
optimizer = optimizer.SGD(model.parameters(), lr=1e-3)
