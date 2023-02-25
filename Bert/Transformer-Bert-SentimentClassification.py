from pathlib import Path
from transformers import Trainer, TrainingArguments
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import torch.optim as optim
import logging

# index = torch.from_numpy(np.array([[1, 2, 0], [2, 0, 1]])).type(torch.LongTensor)
# index = index[:, :, None].expand(-1, -1, 10)
# print(index)

linux_dataset_file_path_train = '/SecondQuestionData/yuchao/Pytorch/Bert/aclImdb/train'
linux_dataset_file_path_test = '/SecondQuestionData/yuchao/Pytorch/Bert/aclImdb/test'
linux_model_file_path = '/SecondQuestionData/yuchao/Model/bert-base-uncase'
linux_logging_file_path = '/SecondQuestionData/yuchao/Pytorch/Bert/Bert_classify1.log'

local_file_train = 'aclImdb1/train'
local_file_test = 'aclImdb1/test'
local_file_model = '../../Model/bert-base-uncase/'
local_logging_file = '../Bert/Bert_classify1.log'


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


device = torch.device('cuda:0')

train_datasets, train_label = load_datasets(linux_dataset_file_path_train)
test_datasets, test_label = load_datasets(linux_dataset_file_path_test)

train_dataset, dev_dataset, train_label, dev_label = train_test_split(train_datasets, train_label, test_size=0.2)    # 这个函数，是用来分割训练集和测试集的

# 读取模型对应的tokenizer
tokenizer = BertTokenizer.from_pretrained(linux_model_file_path)

train_encodings = tokenizer(train_dataset, truncation=True, padding=True, max_length=510)
dev_encodings = tokenizer(dev_dataset, truncation=True, padding=True, max_length=510)
test_encodings = tokenizer(test_datasets, truncation=True, padding=True, max_length=510)


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

        self.model_config = BertConfig.from_pretrained(linux_model_file_path)
        self.model = BertModel.from_pretrained(linux_model_file_path, config=self.model_config)
        # self._do_reinit()

        self.fc.to(device)
        self.softmax.to(device)
        self.model.to(device)

    # def _do_reinit(self):
    #     # Re-init last n layers. 重新微调后面几层
    #     for n in range(2):
    #         self.model.encoder.layer[-(n + 1)].apply(self._init_weight_and_bias)
    #
    # def _init_weight_and_bias(self, module):    # 初始化参数
    #     if isinstance(module, nn.Linear):
    #         module.weight.SecondQuestionData.normal_(mean=0.0, std=self.model_config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.SecondQuestionData.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.SecondQuestionData.zero_()
    #         module.weight.SecondQuestionData.fill_(1.0)]

    def forward(self, input_ids, attention_mask):
        self.model.train()
        outputs = self.model(input_ids, attention_mask)    # 把参数放入

        loss = outputs[0][:, 0, :]    # 取第个句子中第一个cls做文本分类
        loss = self.softmax(self.fc(loss))

        return loss


# 载入模型
model = Bert()
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()


logging.basicConfig(filename=linux_logging_file_path, level=logging.INFO)
logging.info("start")


# 使用原生pytorch微调模型
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

        loss_log = loss.to('cpu')
        str = "loss: ", loss_log.item()
        print(str)
        logging.info(str)

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
    str = "真实标签： ", labels, " 预测标签： ", predict, " 正确率：", right/statistic.size(0)
    logging.info(str)


# args = TrainingArguments(
#     'out',
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     learning_rate=5e-5,
#     evaluation_strategy='epoch',
#     num_train_epochs=3,
#     load_best_model_at_end=True,
# )
#
#
# trainer = Trainer(
#     model,
#     args=args,
#     train_dataset=train_datasets,
#     eval_dataset=test_datasets,
#     tokenizer=tokenizer
# )
#
# trainer.train()
