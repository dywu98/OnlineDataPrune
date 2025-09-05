This is the "roughcraft" version of implementation of ICCV2025 paper "Partial Forward Blocking: A Novel Data Pruning Paradigm for Lossless Training Acceleration".

Currently, this repo only contains the code to train Swin-T model on ImageNet-1k.
More codes on CIFAR-10/100 with more networks (such as ResNet, VGG) will be uploaded soon!

## PFB Image classification （RoughCast Swin-T example）

This folder contains reference training scripts for image classification using PFB.

All experiments have been conducted on 4x 4090 GPUs

PFB parameters:

| PFB Parameter            | help  |
| ------------------------ | ------ |
| `--use-online`           | `Use the proposed PFB data pruning strategy`  |
| `--ratio`                | `prune ratio` |
| `--kernel`               | `KDE Kernel Number`  |
| `--kernel-channel`       | `KDE kernel Channel Number`  |
| `--rterm`                | `random term relative ratio`  |
| `--rscale`               | `random term scale value`  |
| `--alpha`                | `Kernel ema decay (1-alpha)`  |
| `--start-end`            | `PFB online pruning start and end epoch`  |


## PFB training
```
torchrun --nproc_per_node=4 train.py\ 
--model swin_t --epochs 300 --batch-size 256 --opt adamw --lr 0.001 --weight-decay 0.05 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 20 --lr-warmup-decay 0.01 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224 --use-online --ratio 0.4 --start-end 16 260
```

## Baseline training
```
torchrun --nproc_per_node=4 train.py\ 
--model $MODEL --epochs 300 --batch-size 256 --opt adamw --lr 0.001 --weight-decay 0.05 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 20 --lr-warmup-decay 0.01 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224
```
Here `$MODEL` is one of `swin_t`, `swin_s` or `swin_b`.
Note that `--val-resize-size` was optimized in a post-training step, see their `Weights` entry for the exact value.

## InfoBatch training
We also implement InfoBatch in `train_unsup.py`.  You can refer to the settings in the appendix of our ICCV paper to train a network with InfoBatch. Please note that the InfoBatch pruning ratio settting is different from most data pruning methods (including our PFB).
