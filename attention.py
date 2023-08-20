import torch
import torch.nn as nn
from math import sqrt, isnan
from torch import Tensor
import torch.nn.functional as F
from data import *


class Attention(nn.Module):
    def __init__(
        self, seqd: int, embd: int, head_size: int, mask: bool = False
    ) -> None:
        super().__init__()

        self.seqd = seqd
        self.embd = embd
        self.head_size = head_size
        self.mask = mask

        self.ql = nn.Linear(embd, head_size, bias=False)
        self.kl = nn.Linear(embd, head_size, bias=False)
        self.vl = nn.Linear(embd, head_size, bias=False)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Lot's of assertions, to make sure everything has the shapes I expect
        batch_size = q.shape[0]
        assert k.shape[0] == v.shape[0] == batch_size

        # seqd = self.seqd # (fixed, block size, or self.seqd as upper bound)
        seqd = q.shape[1]
        assert q.shape[1] == seqd
        assert k.shape[1] == seqd
        assert v.shape[1] == seqd  # (this bound is not required, for cross-attention)

        assert q.shape[2] == self.embd
        assert k.shape[2] == self.embd
        assert v.shape[2] == self.embd

        q = self.ql(q)
        k = self.kl(k)
        v = self.vl(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.head_size)
        assert scores.shape[0] == batch_size
        assert scores.shape[1] == seqd
        assert scores.shape[2] == seqd

        if self.mask:
            scores = scores.masked_fill(
                torch.tril(torch.ones(seqd, seqd)) == 0, float("-inf")
            )
            assert not isnan(scores.sum())

        ret = scores.softmax(1).matmul(v)
        assert ret.shape[0] == batch_size
        assert ret.shape[1] == seqd
        assert ret.shape[2] == self.head_size

        return ret


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        seqd: int,
        embd: int,
        head_size: int,
        nheads: int,
        mask: bool = False,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [Attention(seqd, embd, head_size, mask) for _ in range(nheads)]
        )

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return torch.cat([head(q, k, v) for head in self.heads], dim=-1)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        seqd: int,
        embd: int,
        head_size: int,
        nheads: int,
        mask: bool = False,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(seqd, embd, head_size, nheads, mask)
        self.proj = nn.Linear(head_size * nheads, embd)
        self.ffwd = nn.Sequential(
            nn.Linear(embd, embd * 4),
            nn.ReLU(),
            nn.Linear(embd * 4, embd),
            nn.Dropout(0.1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x2 = self.attn(x, x, x)
        x = self.proj(x2) + x
        x = self.ffwd(x) + x
        return x


# Embd -> Attn -> Linear
class Bruh(nn.Module):
    def __init__(self, seqd: int, embd: int, head_size: int, nlayers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(len(chars), embd)
        self.pos_embed = nn.Embedding(seqd, embd)
        self.decoder = nn.Sequential(
            *[DecoderBlock(seqd, embd, head_size, 8, mask=True) for _ in range(nlayers)]
        )
        self.out = nn.Linear(embd, len(chars))

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x) + self.pos_embed(torch.arange(x.shape[1]))
        x = self.decoder(x)
        return self.out(x)


if __name__ == "__main__":
    print("--- Bruh ---")

    train_iter = 5000

    bruh = Bruh(block_size, 16, 32, 4)

    optimizer = torch.optim.AdamW(bruh.parameters(), lr=0.001)

    for i in range(train_iter):
        optimizer.zero_grad()
        x, y = get_batch()
        x = bruh(x)

        x = x.view(-1, len(chars))
        y = y.view(-1)
        loss = F.cross_entropy(x, y)
        loss.backward()
        optimizer.step()
        if i % 100:
            losses = []
            for _ in range(10):
                x, y = get_batch()
                x = bruh(x)
                x = x.view(-1, len(chars))
                y = y.view(-1)
                loss = F.cross_entropy(x, y)
                losses.append(loss.item())
            print(
                f"Loss: {torch.tensor(losses).mean().item():.2f} - {int(i*100/train_iter)}%",
                end="\r",
            )

    print()

    tokens = torch.tensor([[stoi["."], stoi[" "]]])

    for _ in range(500):
        x = bruh(tokens[:, -block_size:])
        x = x.view(-1, len(chars))
        x = torch.multinomial(F.softmax(x, dim=1), 1)
        tokens = torch.cat([tokens[0], torch.tensor([x[-1]])]).view(1, -1)

    for c in tokens[0]:
        print(itos[int(c.item())], end="")
    print()
