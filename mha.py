import torch
import torch.nn as nn
from math import sqrt, isnan
from torch import Tensor
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.d = d
        self.heads = heads
        self.qw = nn.Linear(d, d * heads)
        self.kw = nn.Linear(d, d * heads)
        self.vw = nn.Linear(d, d * heads)
        self.ow = nn.Linear(d * heads, d)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        assert q.shape[-1] == self.d
        assert k.shape[-1] == self.d
        assert v.shape[-1] == self.d
        s = [-1, self.d, self.heads]
        q = self.qw(q).view(*s)
        k = self.kw(k).view(*s)
        v = self.vw(v).view(*s)

        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.d)

        return self.ow(scores.softmax(1).matmul(v).reshape(-1, self.heads * self.d))


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
        x = self.mha(x, x, x) + x
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
        x = self.mha(x, x, x) + x
        x = self.norm(x)
        x = self.mha(x, enc, enc) + x
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
        self.out = nn.Linear(d, d)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.d
        enc = self.encoder(x)
        dec = self.decoder(x, enc)
        return self.out(dec).softmax(1)


charecters = list(" \nabcdefghijklmnopqrstuvwxyz.,;'?!")


def naive_tokenizer(line: str) -> Tensor:
    line = line.lower()
    ret = []
    for c in line:
        if c in charecters:
            ret.append(float(charecters.index(c)))
        else:
            ret.append(0)
    return torch.tensor(ret)


if __name__ == "__main__":
    # Predict all zeros for random input
    mha = MultiHeadAttention(3, 2)
    print(mha(torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)).shape)
    optimizer = torch.optim.SGD(mha.parameters(), lr=0.1)

    for _ in range(10):
        optimizer.zero_grad()
        input = torch.randn(1, 3)
        out = mha(input, input, input)
        loss = torch.sum(out**2)
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
        loss = torch.sum(out**2)
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
        loss = torch.sum(out**2)
        loss.backward()
        optimizer.step()
        print(float(loss))

    # Open tiny-shakespeare.txt

    print("---- Shakespeare ----")

    with open("tiny-shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [
        float(stoi[c]) for c in s
    ]  # encoder: take a string, output a list of integers
    data: Tensor = torch.tensor(encode(text))

    # data loading
    def get_batch():
        # generate a small batch of data of inputs x and targets y
        ix = torch.randint(len(data) - block_size, (batch_size,))
        a = torch.stack([data[i : i + block_size] for i in ix])
        b = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return a, b

    block_size = 8
    batch_size = 32
    train_iter = 5000
    transformer = Transformer(block_size, 6)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-3)

    file = open("tiny-shakespeare.txt")

    for i in range(train_iter):
        # Accumulate lines
        # lines = []
        # for line in range(batch_size):
        #    txt = file.read(block_size + 1)
        #    if len(txt) < block_size + 1:
        #        file.seek(0)
        #        txt = file.read(block_size + 1)
        #        print("epoch")
        #    lines.append(txt)

        # optimizer.zero_grad()

        # line = torch.stack([naive_tokenizer(line) for line in lines])

        # print(line.shape)

        target, input = get_batch()
        out = transformer(input)
        # print(out.shape, target.shape)
        # out = out.view(-1, len(charecters))
        # target = target.view(-1)
        l = F.cross_entropy(out, target)
        l.backward()
        optimizer.step()
        # print(float(l))
        if isnan(float(l)):
            exit(1)

        print(int(i * 100 / train_iter), "%", end="\r")

    test_line = "You are all resolved rather to die than to famish?"
    print(test_line[:block_size], end=" -> ")
    test_line = naive_tokenizer(test_line[:block_size])

    for n in range(500):
        x = test_line[:block_size]
        out = transformer(x)
        x = torch.argmax(out[0, :])
        print(charecters[x], end="")
        test_line = torch.cat([test_line[1:], torch.tensor([x])])
    print()

    # TODO: Masked attention
    # TODO: Plot losses
    # TODO: Generate text
