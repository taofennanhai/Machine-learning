from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers import pipeline
import torch

tokenizer = BertTokenizer.from_pretrained("../../Model/bert-base-uncase")
config = BertConfig.from_pretrained('../../Model/bert-base-uncase', output_hidden_states=True)
bertmodel = BertModel.from_pretrained('../../Model/bert-base-uncase', config=config)    # 修改参数


sequence = "Not known as chief executive officers until the 1990s, they were known [MASK] General Secretaries."
tokenized_sequence = tokenizer.encode(sequence,  max_length=20, truncation=True, padding='max_length')
print(tokenized_sequence)    # 每个单词进行wordEmbedding

masked_index = tokenized_sequence.index(tokenizer.mask_token_id)

# print("------------------------")
# print("------------------------")


# Create the segments tensors.
segments_ids = [0] * len(tokenized_sequence)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([tokenized_sequence])
segments_tensors = torch.tensor([segments_ids])

model = BertForMaskedLM.from_pretrained('../../Model/bert-base-uncase')
model.eval()
masked_index = tokenized_sequence.index(tokenizer.mask_token_id)
# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)   # 输出每个词最大的概率

predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)
