from transformers import BertTokenizer
from transformers import pipeline


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sequence = "A Titan RTX has 24GB of VRAM"
tokenized_sequence = tokenizer(sequence)
print(tokenized_sequence)    # 每个单词进行wordEmbedding

print("------------------------")
inputs = tokenizer(sequence)    # 每个单词进行编码
encoded_sequence = inputs["input_ids"]
print(encoded_sequence)

print("------------------------")
decoded_sequence = tokenizer.decode(encoded_sequence)    # 每个单词进行解码
print(decoded_sequence)  # [CLS] A Titan RTX has 24GB of VRAM [SEP]


# Attention mask
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence_a = "This is a short sequence."
sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
encoded_sequence_b = tokenizer(sequence_b)["input_ids"]
print(len(encoded_sequence_a), len(encoded_sequence_b))    # 8, 19 两个句子进行编码的获取

padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
print(padded_sequences["input_ids"])     # [[101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]]
print(padded_sequences["attention_mask"])    # [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


# Token Type IDs 句子ID
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence_a = "HuggingFace is based in NYC"
sequence_b = "Where is HuggingFace based?"
encoded_dict = tokenizer(sequence_a, sequence_b)
decoded = tokenizer.decode(encoded_dict["input_ids"])
print(decoded)    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print(encoded_dict['token_type_ids'])    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"  # change to "tf" for got TensorFlow pt是pytorch
)
for key, value in batch.items():
    print(f"{key}: {value.numpy().tolist()}")



