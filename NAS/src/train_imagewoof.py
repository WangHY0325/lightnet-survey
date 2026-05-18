from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model_registry import MODEL_SPECS, create_model, env_snapshot, get_spec, parse_model_keys


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


def build_loaders(data_root: Path, image_size: int, batch_size: int, workers: int):
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected train/val folders under {data_root}")
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = datasets.ImageFolder(train_dir, train_tf)
    val_ds = datasets.ImageFolder(val_dir, val_tf)
    kwargs = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader, train_ds.classes


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


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, max_batches=None):
    model.train()
    total_loss = 0.0
    total = 0
    iterator = tqdm(loader, desc=f"train e{epoch}", leave=False, disable=not sys.stderr.isatty())
    for step, (images, targets) in enumerate(iterator, start=1):
        if max_batches and step > max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch} step={step}: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        iterator.set_postfix(loss=total_loss / max(total, 1))
    return {"loss": total_loss / max(total, 1), "samples": total}


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total = 0
    top1 = 0.0
    top5 = 0.0
    iterator = tqdm(loader, desc="val", leave=False, disable=not sys.stderr.isatty())
    for step, (images, targets) in enumerate(iterator, start=1):
        if max_batches and step > max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        c1, c5 = accuracy(logits, targets, topk=(1, 5))
        total_loss += loss.item() * images.size(0)
        top1 += c1
        top5 += c5
        total += images.size(0)
    return {
        "loss": total_loss / max(total, 1),
        "top1": 100.0 * top1 / max(total, 1),
        "top5": 100.0 * top5 / max(total, 1),
        "samples": total,
    }


def make_scheduler(optimizer, epochs: int, warmup_epochs: int):
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=epochs)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def record_failure(args, model_key: str, exc: Exception) -> None:
    spec = get_spec(model_key)
    out_dir = args.output_dir / model_key / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_key": model_key,
        "display_name": spec.display_name,
        "source": spec.source,
        "model_name": spec.model_name,
        "paper_family": spec.paper_family,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": "",
        "best_top1": "",
        "best_top5": "",
        "best_loss": "",
        "elapsed_sec": "",
        "checkpoint": "",
        "status": "failed",
        "error": repr(exc),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_csv(args.output_dir / "results.csv", summary)
    print(f"[failed] {spec.display_name}: {exc!r}", flush=True)


def run_model(args, model_key: str, train_loader, val_loader, num_classes: int, device: torch.device):
    spec = get_spec(model_key)
    out_dir = args.output_dir / model_key / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[start] {spec.display_name} ({spec.model_name}) seed={args.seed}", flush=True)
    model = create_model(model_key, num_classes=num_classes, pretrained=not args.no_pretrained)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = GradScaler(enabled=device.type == "cuda")
    best = {"top1": -math.inf, "epoch": 0}
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args.max_train_batches
        )
        val_metrics = evaluate(model, val_loader, criterion, device, args.max_val_batches)
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if val_metrics["top1"] > best["top1"]:
            best = {"epoch": epoch, **val_metrics}
            ckpt = {
                "model_key": model_key,
                "spec": spec.__dict__,
                "classes": train_loader.dataset.classes,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "metrics": best,
                "args": json_safe(vars(args)),
                "env": env_snapshot(),
            }
            torch.save(ckpt, out_dir / "best.pt")
    elapsed = time.time() - started
    summary = {
        "model_key": model_key,
        "display_name": spec.display_name,
        "source": spec.source,
        "model_name": spec.model_name,
        "paper_family": spec.paper_family,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best["epoch"],
        "best_top1": best["top1"],
        "best_top5": best["top5"],
        "best_loss": best["loss"],
        "elapsed_sec": elapsed,
        "checkpoint": str(out_dir / "best.pt"),
        "status": "ok",
        "error": "",
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_csv(args.output_dir / "results.csv", summary)
    print(f"[done] {spec.display_name}: top1={best['top1']:.2f} top5={best['top5']:.2f}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/mobile_nas_imagewoof"))
    parser.add_argument("--models", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train_loader, val_loader, classes = build_loaders(args.data_root, args.image_size, args.batch_size, args.workers)
    manifest = {
        "env": env_snapshot(),
        "data_root": str(args.data_root),
        "classes": classes,
        "num_train": len(train_loader.dataset),
        "num_val": len(val_loader.dataset),
        "args": json_safe(vars(args)),
        "model_specs": [spec.__dict__ for spec in MODEL_SPECS],
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_keys = parse_model_keys(args.models)
    for key in model_keys:
        try:
            run_model(args, key, train_loader, val_loader, len(classes), device)
        except Exception as exc:
            record_failure(args, key, exc)


if __name__ == "__main__":
    main()
