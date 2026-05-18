# Lightnet Survey — Reproduction Code

Reproduction code for the controlled experiments in:

> **A Survey of Lightweight Neural Networks**  
> Hongyang Wang et al.

This repository contains only code we wrote for experiments where the original paper does not provide an official implementation. Methods with official code (CLEAN, SaLoRA, QuEST, MambaIRv2) are not included here.

## Repository Structure

```
├── NAS/src/                  # Table III: Mobile NAS case study (ImageWoof2-160)
├── KD/src/                   # Table IV: Mobile KD case study
├── LowRank/src/              # Table VIII: Pointwise SVD stress/recovery
├── Quant/src/                # Table V: Mobile quantization case
└── lowrank_vision/
    ├── LRPRNet/              # Table VII: LRPRNet reproduction + ablation
    └── AATucker/VGG19/       # Table IX: AATucker VGG-19/CIFAR-10
```

## Training Configuration

All ImageWoof2-160 experiments use:

| Parameter | Value |
|-----------|-------|
| Dataset | ImageWoof2-160 (9025 train / 3929 val, 10 classes) |
| Optimizer | SGD (momentum 0.9, weight decay 4e-5) |
| Learning rate | 0.05, cosine annealing |
| Epochs | 250 |
| Batch size | 64 |
| Input size | 160x160 |
| Seed | 42 |
| GPU | NVIDIA L40 / A30 / A40 |

### LRPRNet (Table VII)

- Width multiplier: 1.0
- Inner rank ratio: 0.125 (r = m/8)
- Ablation variants: `--variant full|no_l2|no_res|baseline`

```bash
python train_ablation.py --variant full --epochs 250 --batch_size 64 --lr 0.05 --seed 42
python train_ablation.py --variant no_l2 --epochs 250 --batch_size 64 --lr 0.05 --seed 42
python train_ablation.py --variant no_res --epochs 250 --batch_size 64 --lr 0.05 --seed 42
python train_ablation.py --variant baseline --epochs 250 --batch_size 64 --lr 0.05 --seed 42
```

### AATucker (Table IX)

- Backbone: VGG-19 on CIFAR-10
- Lambda: 0.0019
- Alternating SGD-based network training and Tucker-rank optimization

```bash
python VGG-CIFAR.py
```

### NAS / KD / Quant / LowRank Case Studies

Each module has a `src/` directory with training, evaluation, and summarization scripts. See the script docstrings for usage.

## Data

- **ImageWoof2-160**: Download from https://github.com/fastai/imagenette
- **CIFAR-10**: Auto-downloaded by torchvision
- **KITTI**: See OpenPCDet documentation

## Citation

If you use this code, please cite our survey paper.

## License

MIT
