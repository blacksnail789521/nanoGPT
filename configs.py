class Bigram_configs(object):
    # Lookup table
    def __init__(self) -> None:
        self.batch_size = 1024
        self.lr = 1e-2

        self.block_size = 2


class NanoGPT_v1_configs(object):
    # Single-head self attention
    def __init__(self) -> None:
        self.batch_size = 1024
        self.lr = 1e-3

        self.block_size = 8
        self.n_embd = 32


class NanoGPT_v2_configs(object):
    # Multi-head self attention
    def __init__(self) -> None:
        self.batch_size = 1024
        self.lr = 1e-3

        self.block_size = 8
        self.n_embd = 128  # head_size = 128 // 4 = 32
        self.n_head = 4


class NanoGPT_v3_configs(object):
    # Blocks (Residual + FF)
    def __init__(self) -> None:
        self.batch_size = 1024
        self.lr = 1e-3

        self.block_size = 8
        self.n_embd = 128  # head_size = 128 // 4 = 32
        self.n_head = 4
        self.n_layer = 3


class NanoGPT_v4_configs(object):
    # Regularization (Dropout + LN)
    def __init__(self) -> None:
        self.batch_size = 1024
        self.lr = 1e-3

        self.block_size = 8
        self.n_embd = 128  # head_size = 128 // 4 = 32
        self.n_head = 4
        self.n_layer = 3
        self.dropout = 0.2


class NanoGPT_v4_scaled_configs(object):
    # Larger, deeper
    def __init__(self) -> None:
        self.batch_size = 64
        self.lr = 1e-4

        self.block_size = 256
        self.n_embd = 384  # head_size = 384 // 6 = 64
        self.n_head = 6
        self.n_layer = 6
        self.dropout = 0.2
