import torch
import torch.nn as nn
import pytorch_lightning as pl

from base import BaseModel


class Bigram(BaseModel):
    def __init__(self, vocab_size: int, learning_rate: float, **kwargs) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_embedding_table(x)  # (B, T) --> (B, T, V)
