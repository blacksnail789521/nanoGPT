import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os
import multiprocessing


class TinyShakespeareDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def load_data() -> tuple[torch.Tensor, int, callable, callable]:  # type: ignore
    # Load raw text data
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Get unique chars
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Tokenize text (change from list to tensor)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    encode = lambda string: [stoi[c] for c in string]
    decode = lambda int_list: "".join([itos[i] for i in int_list])
    data = torch.tensor(encode(text), dtype=torch.long)

    return data, vocab_size, encode, decode


def get_dl(
    data: torch.Tensor, block_size: int, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    # Split data into train and test
    n = int(0.9 * len(data))
    train_data, test_data = data[:n], data[n:]

    # Get Dataset
    train_ds = TinyShakespeareDataset(train_data, block_size)
    test_ds = TinyShakespeareDataset(test_data, block_size)

    # Get DataLoader
    dl_config = {
        "batch_size": batch_size,
        "num_workers": multiprocessing.cpu_count(),
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": True,
    }
    train_dl = DataLoader(train_ds, shuffle=True, **dl_config)
    test_dl = DataLoader(test_ds, shuffle=False, **dl_config)

    return train_dl, test_dl


if __name__ == "__main__":
    block_size = 8
    batch_size = 1024

    # Set all random seeds (Python, NumPy, PyTorch)
    pl.seed_everything(seed=0)

    # Load data
    data, vocab_size, encode, decode = load_data()

    # Get DataLoader
    train_dl, test_dl = get_dl(data, block_size, batch_size)
