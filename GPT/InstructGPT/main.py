import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
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


text = "My name is Merve and my favorite"


encoded_input = tokenizer(text, return_tensors='pt')

labels = torch.LongTensor(np.array([1438, 318, 337, 3760, 290, 616, 4004, 8]))

outputs = gpt2(
        input_ids=encoded_input.input_ids,
        labels=labels
    )

print(tokenizer.pad_token)


