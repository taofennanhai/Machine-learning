import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.optim as optim

# # 读取模型对应的tokenizer
# tokenizer = BertTokenizer.from_pretrained('C:/Users/Administrator/Desktop/Model/bert-base-uncase/')
#
# # 载入模型
# model = BertModel.from_pretrained('C:/Users/Administrator/Desktop/Model/bert-base-uncase/')

# # 输入文本
# input_text = ["Beijing is the Most important capital of China",
#               "Nanchang is the capital of JiangXi Province",
#               "Changsha is the capital of Hunan"]
#
# # 通过tokenizer把文本变成 token_id
# encoder_inputs = tokenizer(input_text)
# # print(tokenizer.decode(encoder_inputs['input_ids']))  # 还原序列化后的句子
# print(encoder_inputs)  # 101和102分别是CLS和SEP
#
# batch = tokenizer(input_text, padding=True, truncation=True, max_length=12)    # 进行填充，进行裁剪
# print(batch)
#
#
# # print(tokenizer.decode(batch['input_ids']))  # 还原序列化后的句子
# print(tokenizer.decode(batch['input_ids'][0]))
# print(tokenizer.decode(batch['input_ids'][1]))
# print(tokenizer.decode(batch['input_ids'][2]))

# tokens_tensor = torch.tensor([input_ids])
#
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# print(output)


# 准备数据集
def load_datasets(split_dir):
    split_dir = Path(split_dir)
    dataset = []
    label = []
    for label_dir in ['pos', 'neg']:                         # 每个文件夹有两个pos,neg文件夹
        for text_file in (split_dir/label_dir).iterdir():    # ‘/’就是链接两个路径，找到pos的文件并且迭代所有文件
            dataset.append(text_file.read_text(encoding='utf-8'))
            label.append(0 if label_dir == 'neg' else 1)

    return dataset, label
    # filenames = [path for path in split_dir.iterdir()]
    # print(filenames)


train_dataset, train_label = load_datasets('aclImdb/train')
test_dataset, test_label = load_datasets('aclImdb/test')

train_dataset, dev_dataset, train_label, dev_label = train_test_split(train_dataset, train_label, test_size=0.2)    # 这个函数，是用来分割训练集和测试集的

# 读取模型对应的tokenizer
tokenizer = BertTokenizer.from_pretrained('C:/Users/Administrator/Desktop/Model/bert-base-uncase/')

train_encodings = tokenizer(train_dataset, truncation=True, padding=True)
dev_encodings = tokenizer(dev_dataset, truncation=True, padding=True)
test_encodings = tokenizer(test_dataset, truncation=True, padding=True)

# # 载入模型
# model = BertModel.from_pretrained('C:/Users/Administrator/Desktop/Model/bert-base-uncase/')


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 建立好可以训练的batch，可以使用Dataloader
train_dataset = IMDbDataset(train_encodings, train_label)
dev_dataset = IMDbDataset(dev_encodings, dev_label)
test_dataset = IMDbDataset(test_encodings, test_label)


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.fc = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=-1)

        self.model = BertModel.from_pretrained('C:/Users/Administrator/Desktop/Model/bert-base-uncase/', output_hidden_states=True)
        self.model.to(device)


    def forward(self, input_ids, attention_mask):
        self.model.train()
        outputs = self.model(input_ids, attention_mask)    # 把参数放入

        loss = outputs[0][:, 0, :]    # 取第个句子中第一个cls做文本分类
        loss = self.softmax(self.fc(loss))

        return loss


device = torch.device('cpu')
# 载入模型
model = Bert()
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()


# 训练
for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        print("loss: ", loss.item())

model.eval()


test_dataset = DataLoader(test_dataset, batch_size=4, shuffle=True)
right = 0
# 预测

for batch in test_dataset:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    outputs = model(input_ids, attention_mask)

    predict = torch.max(outputs, dim=1).indices

    statistic = predict.eq(labels)    # 判断是不是每个都一样
    right = 0
    for elem in statistic:
        if elem.item() is True:
            right += 1
    print("真实标签： ", labels, " 预测标签： ", predict, " 正确率：", right/statistic.size(0))




