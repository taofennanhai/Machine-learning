import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import numpy as np
import os


output_dir = 'model'
output_prefix = 'GPT2_SongLyric'
state_dict = torch.load(os.path.join(output_dir, f"{output_prefix}-{2}.pt"))['state_dict']

entry_length = 50
top_p = 0.8
temperature = 1.0

model = GPT2LMHeadModel.from_pretrained('../../../Model/GPT2-base')
model.load_state_dict(state_dict)

tokenizer = GPT2Tokenizer.from_pretrained('../../../Model/GPT2-base')

prompt = " Lyric of '" + 'My Love' + "' ==>> "

generated = torch.tensor(tokenizer.encode(prompt), device=model.device).unsqueeze(0)

model.eval()
nstart = generated.shape[-1]

entry_finished = False
filter_value = -float("Inf")

for nth in range(entry_length):

    # outputs = model(generated, labels=generated)
    loss, logits, __ = model(generated, labels=generated).to_tuple()

    logits = logits[:, -1, :] / temperature    # 获得最后一个预测词的logit概率

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)    # 逆序排序获得下一个词的index

    test = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)    # 前面和值加上后面的值

    sorted_indices_to_remove = cumulative_probs > top_p    # 当为False的时候就选择，为True部分舍弃

    test1 = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()    # 数组往后挪一位
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices[sorted_indices_to_remove]    # 需要移除的所有index值
    logits[:, indices_to_remove] = filter_value    # 需要移除的位置logits数据定义为inf

    next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)    # 输入的张量作为权重进行多分布采样
    generated = torch.cat((generated, next_token), dim=1)    # 把下一个值结合起来继续预测

    # Flag whether or not the next token is the end-to-string special token
    entry_finished = (next_token.item() == tokenizer.eos_token_id)

    # stop early if end-of-sequence token is reached:
    if entry_finished:
        break

ngenerated = (generated.shape[-1] - nstart)
assert ngenerated == (nth + 1), "sanity check failed; check loop"

output_list = list(generated.cpu().squeeze().numpy())
# output_text = f"{tokenizer.decode(output_list)}{'' if entry_finished else '<|endoftext|>'}"

### only return the new (generated) text:
generated_list = output_list[-ngenerated:]
generated_text = f"{tokenizer.decode(generated_list)}{'' if entry_finished else tokenizer.eos_token}"

print(prompt + generated_text)