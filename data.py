# Stole this code from https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py

import torch
from torch import Tensor

block_size = 14
batch_size = 4

with open("tiny-shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
data = torch.tensor(encode(text))


# data loading
def get_batch():
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    a = torch.stack([data[i : i + block_size] for i in ix])
    b = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return a, b
