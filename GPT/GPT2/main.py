import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import numpy as np
import os
from Load_Dataset import SongLyrics


def pack_tensor(new_tensor, packed_tensor, max_seq_len):   # Accumulated batch size (since GPT2 is so big)
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


tokenizer = GPT2Tokenizer.from_pretrained('../../../Model/GPT2-base')    # Get the tokenizer and model
model = GPT2LMHeadModel.from_pretrained('../../../Model/GPT2-base')
train_dataset = SongLyrics(None, truncate=True, gpt2_type="gpt2")


acc_steps = 100
device = torch.device("cuda")
batch_size = 16
epochs = 5
lr = 2e-5
max_seq_len = 400
warmup_steps = 200
output_dir = 'model'
output_prefix = 'GPT2_SongLyric'

save_model_on_epoch = True

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

loss = 0
accumulating_batch_count = 0
input_tensor = None

model = model.to(device)
model.train()


for epoch in range(epochs):

    print(f"Training epoch {epoch}")
    print(loss)

    # (optional) vector of loss values per minibatch
    losses = []

    for batch_idx, (idx, entry) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), mininterval=15, maxinterval=60, miniters=200, leave=False):
        (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

        if carry_on and idx != len(train_dataloader) - 1:
            continue

        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor, labels=input_tensor)
        loss = outputs[0]
        loss.backward()

        if (((batch_idx + 1) % batch_size) == 0) or ((batch_idx + 1) == len(train_dataloader)):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # input_tensor = None
        input_tensor = remainder
        losses.append(loss.detach().cpu().item())

    print(f"avg loss: {np.mean(losses)} for epoch {epoch}")

    if save_model_on_epoch:
        print('saving epoch state')
        torch.save({
            "epoch": epoch,
            "accum_batches": batch_size,
            "lr": lr,
            "max_seq_len": max_seq_len,
            "state_dict": model.state_dict(),
            "losses": losses,
            # 'accumulate': accumulate
        }, os.path.join(output_dir, f"{output_prefix}-{epoch}.pt")
        )














