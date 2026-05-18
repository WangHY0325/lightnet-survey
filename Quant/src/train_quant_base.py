from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from quant_common import (
    append_row,
    build_loaders,
    env_snapshot,
    evaluate,
    json_safe,
    make_mobilenetv2,
    set_seed,
    write_json,
)


def make_scheduler(optimizer, epochs: int, warmup_epochs: int):
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=epochs)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, max_batches: int = 0):
    model.train()
    total_loss = 0.0
    total = 0
    for step, (images, targets) in enumerate(loader, start=1):
        if max_batches and step > max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step={step}: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return {"loss": total_loss / max(total, 1), "samples": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/quant_imagewoof"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--pretrained-file", default="mobilenet_v2-7ebf99e0.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, _, val_loader, classes = build_loaders(args.data_root, args.image_size, args.batch_size, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_path = None if args.no_pretrained else args.weights_dir / args.pretrained_file
    model = make_mobilenetv2(num_classes=len(classes), pretrained_path=pretrained_path).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    out_dir = args.output_dir / "fp32_baseline" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"top1": -math.inf, "epoch": 0}
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.max_train_batches
        )
        val_metrics = evaluate(model, val_loader, device, max_batches=args.max_val_batches)
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
            torch.save(
                {
                    "model_key": "torchvision_mobilenet_v2",
                    "classes": classes,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "metrics": best,
                    "args": json_safe(vars(args)),
                    "env": env_snapshot(),
                },
                out_dir / "best.pt",
            )

    elapsed = time.time() - started
    summary = {
        "method": "fp32_cuda",
        "label": "FP32 CUDA baseline",
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best["epoch"],
        "best_top1": best["top1"],
        "best_top5": best["top5"],
        "elapsed_sec": elapsed,
        "checkpoint": str(out_dir / "best.pt"),
        "status": "ok",
        "error": "",
        **{f"env_{k}": v for k, v in env_snapshot().items()},
    }
    write_json(out_dir / "summary.json", summary)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    append_row(args.output_dir / "train_results.csv", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
