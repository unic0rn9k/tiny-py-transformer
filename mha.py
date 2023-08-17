import torch
import torch.nn as nn
from math import sqrt
from torch import Tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.d = d
        self.heads = heads
        self.qw = nn.Linear(d, d*heads)
        self.kw = nn.Linear(d, d*heads)
        self.vw = nn.Linear(d, d*heads)
        self.ow = nn.Linear(d*heads, d)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor: # returns [heads, seq_len, d]
        assert q.shape[-1] == self.d
        assert k.shape[-1] == self.d
        assert v.shape[-1] == self.d
        s = [self.heads, -1, self.d]
        q = self.qw(q).view(*s)
        k = self.kw(k).view(*s)
        v = self.vw(v).view(*s)

        return self.ow((torch.matmul(q, k.transpose(-1, -2)) / sqrt(self.d)).softmax(-1).matmul(v).reshape(-1, self.heads * self.d))

class Encoder(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.d = d
        self.heads = heads
        self.mha = MultiHeadAttention(d, heads)
        self.linear = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.d
        x = self.mha(x,x,x) + x
        x = self.norm(x)
        x = self.linear(x) + x
        x = self.norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.d = d
        self.heads = heads
        self.mha = MultiHeadAttention(d, heads)
        self.linear = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: Tensor, enc: Tensor) -> Tensor:
        assert x.shape[-1] == self.d
        assert enc.shape[-1] == self.d
        x = self.mha(x,x,x) + x
        x = self.norm(x)
        x = self.mha(x,enc,enc) + x
        x = self.norm(x)
        x = self.linear(x) + x
        x = self.norm(x)
        return x

class Transformer(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.d = d
        self.heads = heads
        self.encoder = Encoder(d, heads)
        self.decoder = Decoder(d, heads)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.d
        enc = self.encoder(x)
        dec = self.decoder(x, enc)
        return dec



charecters = list("abcdefghijklmnopqrstuvwxyz .,;'")

def onehot(x: int, n: int) -> Tensor:
    return torch.eye(n)[x]

def naive_tokenizer(txt: str) -> Tensor:
    txt = txt.lower()
    ret = []
    for c in txt:
        if c in charecters:
            c = onehot(charecters.index(c), len(charecters))
        else:
            c = onehot(charecters.index(" "), len(charecters))
        ret.append(c)

    return torch.stack(ret)

if __name__ == "__main__":
    # Predict all zeros for random input
    mha = MultiHeadAttention(3, 2)
    print(mha(torch.randn(4, 3),torch.randn(4, 3),torch.randn(4, 3)).shape)
    optimizer = torch.optim.SGD(mha.parameters(), lr=0.1)

    for _ in range(10):
        optimizer.zero_grad()
        input = torch.randn(1, 3)
        out = mha(input,input,input)
        loss = torch.sum(out ** 2)
        loss.backward()
        optimizer.step()
        print(float(loss))

    # And then the same with the encoder...

    print("---- Encoder ----")

    encoder = Encoder(3, 2)
    print("heads: ", encoder.d)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)

    for _ in range(10):
        optimizer.zero_grad()
        input = torch.randn(1, 3)
        out = encoder(input)
        loss = torch.sum(out ** 2)
        loss.backward()
        optimizer.step()
        print(float(loss))


    # And then the same with a transformer...

    print("---- Transformer ----")

    transformer = Transformer(3, 2)
    optimizer = torch.optim.SGD(transformer.parameters(), lr=0.1)

    for _ in range(10):
        optimizer.zero_grad()
        input = torch.randn(1, 3)
        out = transformer(input)
        loss = torch.sum(out ** 2)
        loss.backward()
        optimizer.step()
        print(float(loss))


    # Open tiny-shakespeare.txt

    transformer = Transformer(len(charecters), 8)
    optimizer = torch.optim.SGD(transformer.parameters(), lr=0.1)
    loss = nn.CrossEntropyLoss()
    losses = []

    for line in open("tiny-shakespeare.txt"):
        optimizer.zero_grad()

        line = naive_tokenizer(line)
        out = transformer(line)
        loss = loss(out, line.argmax(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        print(float(loss))
        loss = nn.CrossEntropyLoss()

    # TODO: Masked attention
    # TODO: Plot losses
    # TODO: Generate text

