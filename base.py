import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from abc import ABC, abstractmethod


class BaseModel(pl.LightningModule, ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], mode: str
    ) -> torch.Tensor:
        x, y = batch
        # Get outputs
        y_pred = self(x)

        # Compute loss
        loss = self.loss(y_pred.transpose(1, 2), y)

        # Logging
        sync_dist = False if mode == "train" else True
        log_config = {"sync_dist": sync_dist, "prog_bar": True, "on_epoch": True}
        self.log(f"{mode}_loss", loss, **log_config)

        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.shared_step(batch, "val")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.shared_step(batch, "test")

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x = batch[0]
        x_generated = x.clone()
        # x is (B, T) array of indices in the current context
        for _ in range(self.predict_kwargs["max_new_tokens"]):
            # crop the context to the last block_size tokens
            x_context = x_generated[:, -self.predict_kwargs["block_size"]:]  # (B, T)
            # get the predictions
            y_pred = self(x_context)  # (B, T, V)
            # focus only on the last time step (we don't care about the history)
            y_pred = y_pred[:, -1, :]  # becomes (B, V)
            # apply softmax to get probabilities
            probs = F.softmax(y_pred, dim=-1)  # (B, V)
            # sample from the distribution
            x_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            x_generated = torch.cat((x_generated, x_next), dim=1)  # (B, T+1)

        return x_generated
