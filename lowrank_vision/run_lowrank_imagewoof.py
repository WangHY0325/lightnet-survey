from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from lowrank_common import (
    build_loaders,
    count_params,
    create_mobilenetv2,
    env_snapshot,
    estimate_macs,
    evaluate,
    json_safe,
    measure_latency,
    recalibrate_bn,
    set_seed,
    write_json,
    write_rows,
)


METHODS = ("baseline", "svd075", "svd050", "svd025", "svd050_recovery",
           "svd075_recal", "svd050_recal", "svd025_recal")
METHOD_INFO = {
    "baseline": ("Baseline MobileNetV2", "none"),
    "svd075": ("SVD-1x1 direct", "r=0.75"),
    "svd050": ("SVD-1x1 direct", "r=0.50"),
    "svd025": ("SVD-1x1 direct", "r=0.25"),
    "svd050_recovery": ("SVD-1x1 + recovery", "r=0.50 + 10 epochs"),
    "svd075_recal": ("SVD-1x1 + BN recal", "r=0.75 + BN recal"),
    "svd050_recal": ("SVD-1x1 + BN recal", "r=0.50 + BN recal"),
    "svd025_recal": ("SVD-1x1 + BN recal", "r=0.25 + BN recal"),
}
RATIOS = {"svd075": 0.75, "svd050": 0.50, "svd025": 0.25, "svd050_recovery": 0.50,
          "svd075_recal": 0.75, "svd050_recal": 0.50, "svd025_recal": 0.25}


def parse_methods(raw: str):
    if raw.lower() == "all":
        return list(METHODS)
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    missing = [method for method in methods if method not in METHODS]
    if missing:
        raise ValueError(f"Unknown methods {missing}; known methods: {list(METHODS)}")
    return methods


def decompose_pointwise_conv(conv: nn.Conv2d, ratio: float) -> nn.Sequential:
    if conv.kernel_size != (1, 1) or conv.groups != 1:
        raise ValueError("Only dense 1x1 Conv2d layers are eligible for SVD")
    weight = conv.weight.detach().float().reshape(conv.out_channels, conv.in_channels)
    max_rank = min(conv.out_channels, conv.in_channels)
    rank = max(1, min(max_rank, int(round(max_rank * ratio))))
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    sqrt_s = torch.sqrt(s[:rank])
    first_weight = (torch.diag(sqrt_s) @ vh[:rank, :]).reshape(rank, conv.in_channels, 1, 1)
    second_weight = (u[:, :rank] @ torch.diag(sqrt_s)).reshape(conv.out_channels, rank, 1, 1)

    first = nn.Conv2d(conv.in_channels, rank, kernel_size=1, stride=conv.stride, padding=0, dilation=1, bias=False)
    second = nn.Conv2d(conv.out_channels if False else rank, conv.out_channels, kernel_size=1, stride=1, padding=0, bias=conv.bias is not None)
    first.weight.data.copy_(first_weight.to(first.weight.dtype))
    second.weight.data.copy_(second_weight.to(second.weight.dtype))
    if conv.bias is not None:
        second.bias.data.copy_(conv.bias.detach())
    return nn.Sequential(first, second)


def replace_pointwise_convs(module: nn.Module, ratio: float, prefix: str = "") -> list[dict]:
    records = []
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1) and child.groups == 1:
            old_params = child.weight.numel() + (child.bias.numel() if child.bias is not None else 0)
            replacement = decompose_pointwise_conv(child, ratio)
            new_params = count_params(replacement)
            setattr(module, name, replacement)
            records.append(
                {
                    "name": full_name,
                    "in_channels": child.in_channels,
                    "out_channels": child.out_channels,
                    "old_params": old_params,
                    "new_params": new_params,
                }
            )
        else:
            records.extend(replace_pointwise_convs(child, ratio, full_name))
    return records


def make_scheduler(optimizer, epochs: int, warmup_epochs: int):
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=epochs)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def train_recovery(model, train_loader, val_loader, args, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.recovery_epochs, min(args.warmup_epochs, args.recovery_epochs))
    scaler = GradScaler(enabled=device.type == "cuda")
    best = {"top1": -math.inf, "epoch": 0}
    history = []
    for epoch in range(1, args.recovery_epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for step, (images, targets) in enumerate(train_loader, start=1):
            if args.max_train_batches and step > args.max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite recovery loss at epoch={epoch} step={step}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
        val_metrics = evaluate(model, val_loader, device=device, max_batches=args.max_val_batches)
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_loss / max(total, 1),
            "train_samples": total,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if val_metrics["top1"] > best["top1"]:
            best = {"epoch": epoch, **val_metrics}
    return best, history


def failed_rows(method: str, rank_setting: str, exc: Exception):
    label, _ = METHOD_INFO[method]
    env = env_snapshot()
    result = {
        "method": method,
        "label": label,
        "rank_setting": rank_setting,
        "top1": "",
        "top5": "",
        "samples": "",
        "params": "",
        "macs": "",
        "checkpoint": "",
        "status": "failed",
        "error": repr(exc),
        **{f"env_{k}": v for k, v in env.items()},
    }
    latency = {
        "method": method,
        "batch_size": "",
        "input_size": "",
        "latency_mean_ms": "",
        "latency_p50_ms": "",
        "latency_p90_ms": "",
        "latency_min_ms": "",
        "latency_max_ms": "",
        "status": "failed",
        "error": repr(exc),
        **{f"env_{k}": v for k, v in env.items()},
    }
    return result, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/lowrank_imagewoof"))
    parser.add_argument("--methods", default="all")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recovery-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--latency-batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=500)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, classes = build_loaders(args.data_root, args.image_size, args.batch_size, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = create_mobilenetv2(len(classes), args.checkpoint, device=device).eval()
    methods = parse_methods(args.methods)
    results = []
    latency_rows = []
    decomposition_manifest = {}

    for method in methods:
        label, rank_setting = METHOD_INFO[method]
        print(f"[start] lowrank method={method} rank={rank_setting}", flush=True)
        try:
            model = copy.deepcopy(base_model).to(device).eval()
            ckpt_path = ""
            recovery_history = []
            recovery_best = {}
            if method != "baseline":
                records = replace_pointwise_convs(model, RATIOS[method])
                decomposition_manifest[method] = records
                model.to(device)
                if method.endswith("_recal"):
                    recalibrate_bn(model, train_loader, device, max_batches=20)
                ckpt_path = args.output_dir / method / f"seed{args.seed}" / "decomposed.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                if method.endswith("_recovery"):
                    recovery_best, recovery_history = train_recovery(model, train_loader, val_loader, args, device)
                    ckpt_path = args.output_dir / method / f"seed{args.seed}" / "best.pt"
                torch.save(
                    {
                        "method": method,
                        "rank_setting": rank_setting,
                        "state_dict": model.state_dict(),
                        "source_checkpoint": str(args.checkpoint),
                        "decomposition": records,
                        "recovery_best": recovery_best,
                        "recovery_history": recovery_history,
                        "args": json_safe(vars(args)),
                        "env": env_snapshot(),
                    },
                    ckpt_path,
                )
            metrics = evaluate(model, val_loader, device=device, max_batches=args.max_val_batches)
            macs = estimate_macs(model, args.image_size, device=device)
            latency = measure_latency(
                model,
                device=device,
                input_size=args.image_size,
                batch_size=args.latency_batch_size,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            env = env_snapshot()
            result_row = {
                "method": method,
                "label": label,
                "rank_setting": rank_setting,
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "samples": metrics["samples"],
                "params": count_params(model),
                "macs": macs,
                "checkpoint": str(ckpt_path),
                "status": "ok",
                "error": "",
                **{f"env_{k}": v for k, v in env.items()},
            }
            latency_row = {
                "method": method,
                "batch_size": args.latency_batch_size,
                "input_size": args.image_size,
                **latency,
                "status": "ok",
                "error": "",
                **{f"env_{k}": v for k, v in env.items()},
            }
        except Exception as exc:
            print(f"[failed] lowrank method={method}: {exc!r}", flush=True)
            result_row, latency_row = failed_rows(method, rank_setting, exc)
        results.append(result_row)
        latency_rows.append(latency_row)
        print(json.dumps(result_row, ensure_ascii=False), flush=True)
        print(json.dumps(latency_row, ensure_ascii=False), flush=True)

    write_rows(args.output_dir / "results.csv", results)
    write_rows(args.output_dir / "latency.csv", latency_rows)
    write_json(
        args.output_dir / "decomposition_manifest.json",
        {
            "source_checkpoint": args.checkpoint,
            "methods": methods,
            "decomposition": decomposition_manifest,
            "env": env_snapshot(),
        },
    )


if __name__ == "__main__":
    main()
