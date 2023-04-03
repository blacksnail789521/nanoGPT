import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LightningModuleWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        max_new_tokens: int = 100,
        block_size: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.name = model.__class__.__name__
        self.save_hyperparameters(
            ignore=["model"]
        )  # We can access the hyperparameters via self.hparams

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)  # type: ignore

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
    ) -> None:
        loss = self.shared_step(batch, "val")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss = self.shared_step(batch, "test")

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x = batch[0]
        x_generated = x.clone()
        # x is (B, T) array of indices in the current context
        for _ in range(self.hparams.max_new_tokens):  # type: ignore
            # crop the context to the last block_size tokens
            x_context = x_generated[
                :, -self.hparams.block_size :  # type: ignore
            ]  # (B, T)
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
