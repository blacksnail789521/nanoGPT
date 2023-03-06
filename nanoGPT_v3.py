import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class Head(nn.Module):
    def __init__(self, block_size: int, n_embd: int, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, H)
        q = self.query(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)

        # compute attention scores ("affinities")
        weights = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        out = weights @ v  # (B, T, H)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, block_size: int, n_embd: int, n_head: int) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(block_size, n_embd, head_size) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heads = [head(x) for head in self.heads]  # [(B, T, H), ...]
        out = torch.cat(heads, dim=-1)  # (B, T, C)
        out = self.proj(out)  # (B, T, C)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, block_size: int, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(block_size, n_embd, n_head)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(x)  # (B, T, C)
        x = x + self.ffwd(x)  # (B, T, C)
        return x


class NanoGPT_v3(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        learning_rate: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(block_size, n_embd, n_head) for _ in range(n_layer)]
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)  # (B, T, C), C = n_embd
        pos_emb = self.position_embedding_table(
            torch.arange(
                T, device=self.device
            )  # LightningModule has a property `device`
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C) + (T, C) --> (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, V)

        return logits
