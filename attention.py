import torch
import torch.nn as nn
from math import sqrt, isnan, sin, cos
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
                torch.tril(torch.ones(seqd, seqd)).to(device) == 0, float("-inf")
            )
            assert not isnan(scores.sum())

        ret = F.dropout(scores.softmax(1), 0.2).matmul(v)
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
            nn.Linear(embd, embd * 5),
            nn.ReLU(),
            nn.Linear(embd * 5, embd),
            nn.Dropout(0.1),
        )
        self.ln1 = nn.LayerNorm(embd)
        self.ln2 = nn.LayerNorm(embd)

    def forward(self, x: Tensor) -> Tensor:
        nx = self.ln1(x)
        x2 = self.attn(nx, nx, nx)
        x = F.dropout(self.proj(x2), 0.2) + x
        x = self.ffwd(self.ln2(x)) + x
        return x


def positional_encoding(seqd: int, embd: int) -> Tensor:
    pe = torch.zeros(seqd, embd)
    for pos in range(seqd):
        for i in range(0, embd, 2):
            pe[pos, i] = sin(pos / (10000 ** ((2 * i) / embd)))
            pe[pos, i + 1] = cos(pos / (10000 ** ((2 * (i + 1)) / embd)))
    return pe


# Embd -> Attn -> Linear
class Bruh(nn.Module):
    def __init__(self, seqd: int, embd: int, head_size: int, nlayers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embd)
        self.pos_embd = positional_encoding(seqd, embd).to(device)
        self.decoder = nn.Sequential(
            *[DecoderBlock(seqd, embd, head_size, 6, mask=True) for _ in range(nlayers)]
        )
        self.out = nn.Linear(embd, vocab_size)
        self.ln = nn.LayerNorm(embd)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x) + self.pos_embd[: x.shape[1], :]
        x = self.decoder(x)
        return self.out(self.ln(x))


if __name__ == "__main__":
    print("--- Bruh ---")

    train_iter = 5000

    bruh = Bruh(block_size, 64, 32, 6).to(device)

    optimizer = torch.optim.AdamW(bruh.parameters(), lr=1e-5)

    try:
        bruh.load_state_dict(torch.load("bruh.pt", map_location=torch.device(device)))
        print("Loaded model")
    except FileNotFoundError:
        print("No model found")

    for i in range(train_iter):
        try:
            optimizer.zero_grad()
            x, y = get_batch()
            x = bruh(x)

            x = x.view(-1, vocab_size)
            y = y.view(-1)
            loss = F.cross_entropy(x, y)
            loss.backward()
            optimizer.step()
            if i % 100:

                def get_loss(train):
                    losses = []
                    for _ in range(10):
                        x, y = get_batch(train=train)
                        x = bruh(x)
                        x = x.view(-1, vocab_size)
                        y = y.view(-1)
                        loss = F.cross_entropy(x, y)
                        losses.append(loss.item())
                    return losses

                train_loss = torch.tensor(get_loss(True)).mean().item()
                val_loss = torch.tensor(get_loss(False)).mean().item()
                print(
                    f"Loss: {train_loss:.2f} - Val loss: {val_loss:.2f} - {int(i*100/train_iter)}%",
                    end="\r",
                )

        except KeyboardInterrupt:
            break

    torch.save(bruh.state_dict(), "bruh.pt")

    print("\n--- Generating ---")

    tokens = torch.tensor([[stoi["."]]]).to(device)

    for _ in range(500):
        x = bruh(tokens[:, -block_size:])
        x = x.view(-1, vocab_size)
        x = torch.multinomial(F.softmax(x * 1.4, dim=1), 1)
        tokens = torch.cat([tokens[0], torch.tensor([x[-1]]).to(device)]).view(1, -1)

    for c in tokens[0]:
        # print("|" + itos[int(c.item())], end="")
        print(itos[int(c.item())], end="")
    print()
