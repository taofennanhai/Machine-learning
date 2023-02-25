import numpy as np
import torch
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification, AdamW

x = np.array([[[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]],
              [[9, 15, 10],
               [12, 13, 14],
               [15, 16, 17]],
              [[18, 19, 20],
               [21, 22, 23],
               [24, 25, 26]], ])

print(x.shape)

temp = torch.tensor(x[:, 0, :])

print(temp)
print(torch.argmax(temp, dim=1))

correct = torch.tensor(0)
print(correct)


tokenizer = BertTokenizer.from_pretrained('../../../Model/bert-base-uncase')

sentence = "it's a sentence"
print(tokenizer.encode(sentence))
print(tokenizer.decode([1005, 1055]))

