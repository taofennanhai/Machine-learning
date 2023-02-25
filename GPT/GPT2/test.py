import os
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
import re

tok_delim = re.compile(r'\s+')
SONG_COLS = ['Artist', 'SName']

# ----- Data Prep -----

### Prepare data
lyrics = pd.read_csv('archive/lyrics-data.csv')
lyrics = lyrics[lyrics.language == 'en']

artists = pd.read_csv('archive/artists-data.csv')
artists.loc[:, "Genres"] = artists.Genres.str.split(";")
artists = artists.explode("Genres")
artists.loc[:, "Genres"] = artists.Genres.str.strip()

### Only keep popular artists, with genre Rock/Pop and popularity high enough
artists = artists[(artists['Genres'].isin(['Rock', 'Pop'])) & (artists['Popularity'] > 5)]

### Drop duplicated artist rows (keeping 'Rock' over 'Pop')
artists.sort_values('Genres', ascending=False, inplace=True)
artists.drop_duplicates(subset=list(set(artists.columns) - set(['Genres'])), inplace=True, keep='first')

### Join lyrics, artists
df = lyrics.merge(artists[['Artist', 'Genres', 'Link']], left_on='ALink', right_on='Link', how='inner')
df.drop(columns=['ALink', 'SLink', 'Genres', 'Link'], inplace=True)

### Tokenize lyric text, add columns to df
tmp = df.Lyric.str.split('\s+')


def notempty(y): return (len(y) > 0)


tmp = tmp.apply(lambda x: list(filter(notempty, x)))
lyric_nwords = tmp.apply(len)
df.insert(df.shape[1], 'lyric_nwords', lyric_nwords)
df.insert(df.shape[1], 'lyric_tokens', tmp)

### ... overwrite original lyric strings with simplified versions
df.loc[:, "Lyric"] = df.lyric_tokens.apply(' '.join)

### filter out songs with too few (<25) or too many words (>350)
df = df[(lyric_nwords >= 25) & (lyric_nwords < 350)].reset_index(drop=True)
del lyric_nwords, tmp

### Create a very small test set to compare generated text with the reality
test_set = df.sample(n=200, random_state=106)
train_set = df.drop(index=test_set.index).copy()

test_set.reset_index(drop=True, inplace=True)
train_set.reset_index(drop=True, inplace=True)

### sanity checks
### ... row counts
assert df.shape[0] == (train_set.shape[0] + test_set.shape[0])
### ... confirm no overlapping songs
shared_songs = train_set.loc[:, SONG_COLS].merge(test_set.loc[:, SONG_COLS], how='inner')
assert shared_songs.shape[0] == 0, "ERROR: overlapping songs in test, train sets"

### For the test set only, keep last 20 words in a new column, then remove them from original column
test_set.insert(test_set.shape[1], 'True_end_lyrics', test_set.lyric_tokens.str[-20:].apply(' '.join))
test_set.loc[:, 'Lyric'] = test_set.lyric_tokens.str[:-20].apply(' '.join)


class SongLyrics(Dataset):
    def __init__(self, lyrics: pd.Series, gpt2_type="gpt2", max_length=1022, truncate=0, **kwargs):

        self.tokenizer = GPT2Tokenizer.from_pretrained('../../../Model/GPT2-base', **kwargs)
        self.lyrics = []

        for i, text in lyrics.iteritems():

            if (truncate > 0) and (i == truncate):
                break

            lyric_toks = self.tokenizer.tokenize(text)
            if len(lyric_toks) > max_length:
                istart = np.random.randint((len(lyric_toks) - max_length))
                lyric_toks = lyric_toks[istart:(istart + max_length)]

            self.lyrics.append(torch.tensor([
                self.tokenizer.bos_token_id,
                *self.tokenizer.convert_tokens_to_ids(lyric_toks),
                self.tokenizer.eos_token_id
            ]))

        self.lyrics_count = len(self.lyrics)

    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, idx):
        return idx, self.lyrics[idx]


output_dir = 'model'
output_prefix = 'GPT2_SongLyric'
epoch = 1

test = os.path.join(output_dir, f"{output_prefix}-{epoch}.pt")

print(test)


logits = sorted_indices = torch.Tensor([[i+1 for i in range(50257)]])

sorted_indices_to_remove = torch.Tensor([[i+1 for i in range(50257)]])

test = sorted_indices_to_remove[..., :-1].clone()
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
sorted_indices_to_remove[..., 0] = 0

filter_value = -float("Inf")

# indices_to_remove = sorted_indices[sorted_indices_to_remove]
logits[:, 1] = filter_value    # 移除的logits数据定义为inf

print(test)


torch.cuda.device_count()
# dataset = SongLyrics(train_set.Lyric, gpt2_type="gpt2")
#
# # Get the tokenizer and model
# tokenizer = dataset.tokenizer
# model = GPT2LMHeadModel.from_pretrained('gpt2')