import torch
from torch import Tensor
from bpe import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

block_size = 20
batch_size = 20

# here are all the unique characters that occur in this text
vocab_size = len(stoi)
# create a mapping from characters to integers
itos = [c for c in stoi]
data = torch.tensor([token for token in tokenize(text_data) if type(token) == int]).to(
    device
)
train_data = data[: -len(data) // 10]
val_data = data[-len(data) // 10 :]


# data loading
def get_batch(train=True):
    if train:
        data = train_data
    else:
        data = val_data
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    a = torch.stack([data[i : i + block_size] for i in ix])
    b = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return a, b
