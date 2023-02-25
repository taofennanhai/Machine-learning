import torch
import numpy
import nltk
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import word_tokenize

# nltk.download()

data = ["I love machine learning. Its awesome.",    # 获取数据
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

print(tagged_data)

max_epochs = 100
vec_size = 20    # 20维
alpha = 0.025

# dm=1使用PV-DM模型，dm=0模型使用PV-DBOW
model = Doc2Vec(dm=1, alpha=alpha, min_count=1, )





