import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os

from load_data import load_data, get_dl

from bigram import Bigram
from nanoGPT_v1 import NanoGPT_v1
from nanoGPT_v2 import NanoGPT_v2
from nanoGPT_v3 import NanoGPT_v3
from nanoGPT_v4 import NanoGPT_v4
from configs import (
    Bigram_configs,
    NanoGPT_v1_configs,
    NanoGPT_v2_configs,
    NanoGPT_v3_configs,
    NanoGPT_v4_configs,
    NanoGPT_v4_scaled_configs,
)


def get_configs_and_model(
    model_name: str, vocab_size: int
) -> tuple[dict, pl.LightningModule]:
    if model_name == "bigram":
        configs = Bigram_configs().__dict__
        model = Bigram(vocab_size=vocab_size, **configs)
    elif model_name == "nanoGPT_v1":
        configs = NanoGPT_v1_configs().__dict__
        model = NanoGPT_v1(vocab_size=vocab_size, **configs)
    elif model_name == "nanoGPT_v2":
        configs = NanoGPT_v2_configs().__dict__
        model = NanoGPT_v2(vocab_size=vocab_size, **configs)
    elif model_name == "nanoGPT_v3":
        configs = NanoGPT_v3_configs().__dict__
        model = NanoGPT_v3(vocab_size=vocab_size, **configs)
    elif model_name == "nanoGPT_v4":
        configs = NanoGPT_v4_configs().__dict__
        model = NanoGPT_v4(vocab_size=vocab_size, **configs)
    elif model_name == "nanoGPT_v4_scaled":
        configs = NanoGPT_v4_scaled_configs().__dict__
        model = NanoGPT_v4(vocab_size=vocab_size, **configs)
    else:
        raise ValueError(f"model_name {model_name} is not supported")

    return configs, model


def train_model(
    train_dl: DataLoader,
    test_dl: DataLoader,
    model: pl.LightningModule,
    use_model: str,
    use_gpu: bool = False,
) -> pl.Trainer:
    # Set callbacks
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=-1,  # save all models
        save_weights_only=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=3, verbose=True
    )
    callbacks = [model_checkpoint, early_stopping]

    # Set trainer
    default_root_dir = os.path.join("lightning_logs", use_model)
    os.makedirs(os.path.join(default_root_dir, "lightning_logs"), exist_ok=True)
    device_config = {}
    if not use_gpu:
        device_config["accelerator"] = "cpu"
    else:
        device_config["accelerator"] = "gpu"
        device_config["devices"] = [0, 1, 2, 3]  # We have
        device_config[
            "strategy"
        ] = "ddp_find_unused_parameters_false"  # Allow to have unused parameters
    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        max_epochs=-1,  # infinite training (stop by early stopping)
        log_every_n_steps=50,  # default: 50
        callbacks=callbacks,
        **device_config,
    )

    # Train the model
    trainer.fit(model, train_dl, test_dl)

    return trainer


def test_model(
    model: pl.LightningModule,
    trainer: pl.Trainer,
    test_dl: DataLoader,
) -> dict:
    # Test the model
    loss_list = trainer.test(model, test_dl, ckpt_path="best")

    # The length of the loss_list corresponds to the number of dataloaders used.
    test_loss_dict = loss_list[0]

    return test_loss_dict


def generate_predictions(
    model: pl.LightningModule,
    trainer: pl.Trainer,
    decode: callable,
    block_size: int,
    test_loss: float,
    use_model: str,
    max_new_tokens: int = 100,
) -> torch.tensor:
    # Generate text (using `predict_step`)
    x_given = torch.zeros(
        4, 1, dtype=torch.long
    )  # 0 means \n (we need to specify the batch size for the first dimension)
    zero_dataloader = DataLoader(TensorDataset(x_given), batch_size=1)
    model.predict_kwargs = {"max_new_tokens": max_new_tokens, "block_size": block_size}
    x_generated_list = trainer.predict(model, dataloaders=zero_dataloader)
    x_generated = x_generated_list[0]  # only get the first batch

    # Save the generated text
    text_path = os.path.join(
        "text", f"generated_text_from_{use_model} (loss={test_loss:.4f}).txt"
    )
    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    with open(text_path, "w") as f:
        f.write(decode(x_generated[0].tolist()))

    return x_generated


if __name__ == "__main__":
    """---------------------------------------"""
    # use_gpu = False
    use_gpu = True

    # use_model = "bigram"
    # use_model = "nanoGPT_v1"
    # use_model = "nanoGPT_v2"
    # use_model = "nanoGPT_v3"
    # use_model = "nanoGPT_v4"
    use_model = "nanoGPT_v4_scaled"
    """---------------------------------------"""

    # Set all random seeds (Python, NumPy, PyTorch)
    pl.seed_everything(seed=0)

    # Load data
    data, vocab_size, encode, decode = load_data()

    # Get configs and model
    configs, model = get_configs_and_model(use_model, vocab_size)

    # Get DataLoader
    train_dl, test_dl = get_dl(data, configs["block_size"], configs["batch_size"])

    # Train
    print("---------------------------------------")
    print("Training ...")
    trainer = train_model(train_dl, test_dl, model, use_model, use_gpu)

    # Test
    print("---------------------------------------")
    print("Testing ...")
    test_loss_dict = test_model(model, trainer, test_dl)

    # Predict
    print("---------------------------------------")
    print("Predicting ...")
    generate_predictions(
        model,
        trainer,
        decode,
        configs["block_size"],
        test_loss_dict["test_loss"],
        use_model,
        max_new_tokens=10000,
    )
    print("### Done ###")
