from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers import pipeline
import torch

tokenizer = BertTokenizer.from_pretrained("../../Model/bert-base-uncase")
config = BertConfig.from_pretrained('../../Model/bert-base-uncase', output_hidden_states=True)
bertmodel = BertModel.from_pretrained('../../Model/bert-base-uncase', config=config)    # 修改参数


sequence = "A Titan RTX has 24GB of VRAM"
tokenized_sequence = tokenizer(sequence,  max_length=20, truncation=True, padding='max_length')
print(tokenized_sequence)    # 每个单词进行wordEmbedding


# print("------------------------")
# print("------------------------")
#
decoded_sequence = tokenized_sequence["input_ids"]
decoded_sequence = tokenizer.decode(decoded_sequence)   # 每个单词进行解码
print(decoded_sequence)  # [CLS] A Titan RTX has 24GB of VRAM [SEP]

# print("------------------------")
# inputs = tokenizer(sequence)    # 每个单词进行编码
# encoded_sequence = inputs["input_ids"]
# print(encoded_sequence)

#
# print("------------------------")
# decoded_sequence = tokenizer.decode(encoded_sequence)    # 每个单词进行解码
# print(decoded_sequence)  # [CLS] A Titan RTX has 24GB of VRAM [SEP]
#
#
# # # Attention mask
# # from transformers import BertTokenizer
# # tokenizer = BertTokenizer.from_pretrained("../../Model/bert-base-uncase")
# # sequence_a = "This is a short sequence."
# # sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."
# #
# # encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
# # encoded_sequence_b = tokenizer(sequence_b)["input_ids"]
# # print(len(encoded_sequence_a), len(encoded_sequence_b))    # 8, 19 两个句子进行编码的获取
# #
# # padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
# # print(padded_sequences["input_ids"])     # [[101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]]
# # print(padded_sequences["attention_mask"])    # [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
# #
# #
# # # Token Type IDs 句子ID
# # from transformers import BertTokenizer
# # tokenizer = BertTokenizer.from_pretrained("../../Model/bert-base-uncase")
# # sequence_a = "HuggingFace is based in NYC"
# # sequence_b = "Where is HuggingFace based?"
# # encoded_dict = tokenizer(sequence_a, sequence_b)
# # decoded = tokenizer.decode(encoded_dict["input_ids"])
# # print(decoded)    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# # print(encoded_dict['token_type_ids'])    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# #
# #
# # batch = tokenizer(
# #     ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
# #     padding=True,
# #     truncation=True,
# #     max_length=512,
# #     return_tensors="pt"  # change to "tf" for got TensorFlow pt是pytorch
# # )
# # for key, value in batch.items():
# #     print(f"{key}: {value.numpy().tolist()}")
#
# input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0)
# attention_mask = torch.tensor(inputs["attention_mask"]).unsqueeze(0)
# token_type_ids = torch.tensor(inputs["token_type_ids"]).unsqueeze(0)
#
# outputs = bertmodel(input_ids, attention_mask, token_type_ids)
#
# last_hidden = outputs.last_hidden_state
# pooler_output = outputs.pooler_output
# hidden_states = outputs.hidden_states
#
# len = last_hidden * attention_mask.unsqueeze(-1)
#
# test = ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
#
# last = outputs.last_hidden_state.transpose(1, 2)    # 转置为[batch, hidden_size(768), seq_len]
# # torch.avg_pool1d是把列相加，每kernel_size大小的列相加求平均，然后跳过kernel_size大小的列
# test2.py = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)    # 把最后一个维度去掉， 最后大小为[batch, 768]
#
#
# first_hidden = hidden_states[0]
# last_hidden = hidden_states[-1]
#
# avg_hidden = ((first_hidden + last_hidden)/2).transpose(1, 2)
# avg = torch.avg_pool1d(avg_hidden, kernel_size=avg_hidden.shape[-1]).squeeze(-1)
#
# pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
#
#
# print(outputs)

