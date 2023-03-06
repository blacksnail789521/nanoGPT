# nanoGPT
The simplest, fastest repository for training/finetuning medium-sized GPTs.

|              Method            |                  Details                |     Test Loss    |     # params    |     Model Size    |
|:------------------------------:|:---------------------------------------:|:----------------:|:---------------:|:-----------------:|
|        bigram (baseline)       |               Lookup table              |       2.4844     |       4.2 K     |        17 KB      |
|            nanoGPT_v1          |     Single-head       self attention    |       2.3394     |       7.6 K     |        30 KB      |
|            nanoGPT_v2          |      Multi-head       self attention    |       2.0876     |      83.4 K     |       334 KB      |
|            nanoGPT_v3          |       Blocks       (Residual + FF)      |       1.7943     |       609 K     |       2.4 MB      |
|            nanoGPT_v4          |       Regularization (Dropout + LN)     |       1.7712     |       611 K     |       2.5 MB      |
|     nanoGPT_v4      _scaled    |           Larger,       deeper          |       1.4942     |      10.8 M     |       43.2 MB     |