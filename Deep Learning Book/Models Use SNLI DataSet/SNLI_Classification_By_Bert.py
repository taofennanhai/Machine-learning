import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from d2l import torch as d2l
import Read_SNLI_DataSet
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import torch.optim as optim

device = "cuda:0"   # "cpu"
premises, hypotheses, labels = Read_SNLI_DataSet.Read_Snli(True)    # 加载数据集


class SNLIDataset(nn.Module):

    def __init__(self, premises, hypotheses, labels, max_len=50):
        tokenizer = BertTokenizer.from_pretrained('../../../Model/bert-base-uncase')

        ensembles = []
        for premise, hypothesis in zip(premises, hypotheses):
            str = (premise, hypothesis)
            ensembles.append(str)
        inputs = tokenizer(ensembles, add_special_tokens=True, truncation=True, padding=True, max_length=max_len)  # encode函数中的add_special_tokens设置为False时，同样不会出现开头和结尾标记：[cls]

        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.inputs.items()}
        item['label'] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)


# 建立好可以训练的batch，可以使用Dataloader
train_dataset = SNLIDataset(premises, hypotheses, labels, max_len=50)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

# i = 0
# for item in train_loader:
#     print(item['input_ids'])
#     print(item['label'])
#     print(item['input_ids'].numpy().tolist())
#
#     for strs in item['input_ids'].numpy().tolist():    # 把效果打印出来
#         print(tokenizer.decode(strs))
#
#     i = i+1
#     if i > 3:
#         break


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.fc = nn.Linear(768, 3)    # 三分类
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)

        self.model_config = BertConfig.from_pretrained('../../../Model/bert-base-uncase')    # 模型存放地址
        model = BertModel.from_pretrained('../../../Model/bert-base-uncase', config=self.model_config)

        self.model = model

        self.fc.to(device)
        self.softmax.to(device)
        self.model.to(device)
        self.dropout.to(device)

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)    # 取第个句子中第一个cls做文本分类

        cls_hidden = output[0][:, 0, :]    # 最后一层的cls隐藏向量,维度形状为[batch_size, hidden_dim]

        loss = self.fc(self.dropout(cls_hidden))    # 经过线性转换，维度为[batch_size, 3]
        return loss


model = Bert()
optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)              # 传入token_id，最后会经过embedding送入模型
        attention_mask = batch['attention_mask'].to(device)    # 传入需要mask掉的部分，如pad部分
        token_type_ids = batch['token_type_ids'].to(device)    # 传入segement编号 不传入默认都是0
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)    # 进行交叉熵损失计算    output维度[batch_size, 3]就是三个种类。labels维度为[batch_size]
        loss.backward()
        optimizer.step()

        # loss_log = loss.to('cpu')
        # str = "loss: ", loss_log.item()
        print("epoch:" + str(epoch+1)+" loss: ", end='')
        print(loss.item())

# 模型参数保存
model_state_dict = model.state_dict()
torch.save(model_state_dict, 'model_state_dict.pkl')


# 模型参数读取
model_state_dict_load = torch.load('model_state_dict.pkl')
print(model_state_dict_load.keys())

new_model = Bert()    # 新建立一个模型
new_model.load_state_dict(model_state_dict_load)


model.eval()    # 关闭模型中的dropout和batchnorm
new_model.eval()
with torch.no_grad():
    test_premises, test_hypotheses, test_labels = Read_SNLI_DataSet.Read_Snli(False)  # 加载测试数据集

    # 建立好测试的batch，可以使用Dataloader
    test_dataset = SNLIDataset(test_premises, test_hypotheses, test_labels, max_len=50)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    total_correct = torch.tensor(0)
    new_total_correct = torch.tensor(0)
    for batch in test_loader:

        input_ids = batch['input_ids'].to(device)              # 传入token_id，最后会经过embedding送入模型
        attention_mask = batch['attention_mask'].to(device)    # 传入需要mask掉的部分，如pad部分
        token_type_ids = batch['token_type_ids'].to(device)    # 传入segement编号 不传入默认都是0
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)  # output维度[batch_size, 3]就是三个种类
        new_outputs = new_model(input_ids, attention_mask, token_type_ids)  # output维度[batch_size, 3]就是三个种类

        y_prediction = torch.argmax(outputs, dim=1)    # 找到最大下表索引
        new_y_prediction = torch.argmax(new_outputs, dim=1)  # 找到最大下表索引

        correct = torch.eq(y_prediction, labels).sum()    # 计算批次的正确个数
        new_correct = torch.eq(new_y_prediction, labels).sum()  # 计算批次的正确个数

        total_correct = total_correct + correct
        new_total_correct = new_total_correct + new_correct

        print("批次正确率：", end="")
        print(correct/batch['input_ids'].shape[0], end="")
        print(" 新的批次正确率：", end="")
        print(new_correct / batch['input_ids'].shape[0])

    print("总正确率： ", total_correct/len(test_dataset), end="")
    print(" 新模型总正确率： ", new_total_correct / len(test_dataset))
