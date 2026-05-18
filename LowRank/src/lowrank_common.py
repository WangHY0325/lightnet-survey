from __future__ import annotations

import csv
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def env_snapshot() -> dict:
    info = {
        "torch": torch.__version__,
        "timm": timm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_runtime"] = torch.version.cuda
    return info


def build_loaders(data_root: Path, image_size: int, batch_size: int, workers: int):
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected train/val folders under {data_root}")
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_ds = datasets.ImageFolder(train_dir, train_tf)
    val_ds = datasets.ImageFolder(val_dir, val_tf)
    kwargs = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader, train_ds.classes


def create_mobilenetv2(num_classes: int, checkpoint: Path, device: torch.device) -> nn.Module:
    model = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=num_classes)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    return model.to(device)


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
    out = []
    for k in topk:
        kk = min(k, logits.shape[1])
        out.append(correct[:kk].reshape(-1).float().sum(0).item())
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, max_batches: int = 0):
    model.eval()
    total = 0
    top1 = 0.0
    top5 = 0.0
    for step, (images, targets) in enumerate(loader, start=1):
        if max_batches and step > max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        c1, c5 = accuracy(logits, targets, topk=(1, 5))
        top1 += c1
        top5 += c5
        total += images.size(0)
    return {"top1": 100.0 * top1 / max(total, 1), "top5": 100.0 * top5 / max(total, 1), "samples": total}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_macs(model: nn.Module, input_size: int, device: torch.device):
    try:
        from thop import profile

        x = torch.randn(1, 3, input_size, input_size, device=device)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return macs
    except Exception as exc:
        print(f"[warn] thop failed: {exc}", flush=True)
        return ""


@torch.no_grad()
def measure_latency(model: nn.Module, device: torch.device, input_size: int, batch_size: int, warmup: int, repeats: int):
    model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "latency_mean_ms": statistics.mean(times),
        "latency_p50_ms": statistics.median(times),
        "latency_p90_ms": sorted(times)[int(0.9 * (len(times) - 1))],
        "latency_min_ms": min(times),
        "latency_max_ms": max(times),
    }


@torch.no_grad()
def recalibrate_bn(model: nn.Module, loader, device: torch.device, max_batches: int = 20):
    """Recalibrate BatchNorm running statistics after weight changes (e.g. SVD).
    Runs a forward pass in train() mode so BN layers update their running_mean/var.
    """
    model.train()
    for step, (images, _) in enumerate(loader, start=1):
        if max_batches and step > max_batches:
            break
        model(images.to(device, non_blocking=True))
    model.eval()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
