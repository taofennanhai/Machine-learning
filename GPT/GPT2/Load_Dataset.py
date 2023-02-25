import torch
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer


### Prepare data
lyrics = pd.read_csv('archive/lyrics-data.csv')
lyrics = lyrics[lyrics['language'] == 'en']

artists = pd.read_csv('archive/artists-data.csv')
artists = artists[(artists['Genres'].str.contains('Rock')) & (artists['Popularity'] > 5)]   # Only keep popular artists, with genre Rock/Pop and popularity high enough

df = lyrics.merge(artists[['Artist', 'Genres', 'Link']], left_on='ALink', right_on='Link', how='inner')
df = df.drop(columns=['ALink', 'SLink', 'language', 'Link'])

df = df[df['Lyric'].apply(lambda x: len(x.split(' ')) < 350)]   # Drop the songs with lyrics too long (after more than 1024 tokens, does not work)

test_set = df.sample(n=200)   # Create a very small test set to compare generated text with the reality
df = df.loc[~df.index.isin(test_set.index)]

test_set = test_set.reset_index()   # Reset the indexes
df = df.reset_index()

test_set['True_end_lyrics'] = test_set['Lyric'].str.split().str[-20:].apply(' '.join)    # For the test set only, keep last 20 words in a new column, then remove them from original column
test_set['Lyric'] = test_set['Lyric'].str.split().str[:-20].apply(' '.join)


class SongLyrics(Dataset):
    def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained('../../../Model/GPT2-base')
        self.lyrics = []

        for title, lyric in zip(df['SName'], df['Lyric']):

            alltext = " Lyric of '" + title + "' ==>> " + f"{lyric[:max_length]}<|endoftext|>"

            self.lyrics.append(torch.tensor(
                self.tokenizer.encode(" Lyric of '" + title + "' ==>> " + f"{lyric[:max_length]}<|endoftext|>")
            ))


        if truncate:
            self.lyrics = self.lyrics[:20000]
        self.lyrics_count = len(self.lyrics)

    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        return item, self.lyrics[item]


dataset = SongLyrics(df['Lyric'], truncate=True, gpt2_type="gpt2")




print('')



