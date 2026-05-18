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
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model_utils import STUDENT_SPEC, TEACHER_SPEC, create_student, env_snapshot, forward_features, load_teacher, pooled_features


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
METHODS = ("ce", "kd", "dkd", "fitnet", "rkd")


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


def kd_loss(student_logits, teacher_logits, temperature: float):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)


def dkd_loss(student_logits, teacher_logits, targets, alpha: float, beta: float, temperature: float):
    gt_mask = torch.zeros_like(student_logits, dtype=torch.bool).scatter_(1, targets[:, None], True)
    other_mask = ~gt_mask

    pred_student = F.softmax(student_logits / temperature, dim=1)
    pred_teacher = F.softmax(teacher_logits / temperature, dim=1)
    pred_student_two = torch.cat([(pred_student * gt_mask).sum(1, keepdim=True), (pred_student * other_mask).sum(1, keepdim=True)], dim=1)
    pred_teacher_two = torch.cat([(pred_teacher * gt_mask).sum(1, keepdim=True), (pred_teacher * other_mask).sum(1, keepdim=True)], dim=1)
    tckd = F.kl_div(torch.log(pred_student_two.clamp_min(1e-8)), pred_teacher_two, reduction="batchmean") * (temperature ** 2)

    nckd_logits_student = student_logits / temperature
    nckd_logits_teacher = teacher_logits / temperature
    # Keep the mask finite for fp16 autocast; -1e9 overflows Half.
    nckd_logits_student = nckd_logits_student.masked_fill(gt_mask, -1e4)
    nckd_logits_teacher = nckd_logits_teacher.masked_fill(gt_mask, -1e4)
    nckd = F.kl_div(
        F.log_softmax(nckd_logits_student, dim=1),
        F.softmax(nckd_logits_teacher, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return alpha * tckd + beta * nckd


def fitnet_loss(student_feat, teacher_feat, adapter):
    student_feat = adapter(student_feat)
    if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
        student_feat = F.interpolate(student_feat, size=teacher_feat.shape[-2:], mode="bilinear", align_corners=False)
    student_feat = F.normalize(student_feat.flatten(2), dim=1)
    teacher_feat = F.normalize(teacher_feat.flatten(2), dim=1)
    return F.mse_loss(student_feat, teacher_feat)


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp_min(eps)
    if not squared:
        res = res.sqrt()
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def rkd_loss(student_feat, teacher_feat):
    with torch.no_grad():
        t_d = pdist(teacher_feat, squared=False)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td.clamp_min(1e-8)
    s_d = pdist(student_feat, squared=False)
    mean_sd = s_d[s_d > 0].mean()
    s_d = s_d / mean_sd.clamp_min(1e-8)
    loss_d = F.smooth_l1_loss(s_d, t_d)

    with torch.no_grad():
        td = teacher_feat.unsqueeze(0) - teacher_feat.unsqueeze(1)
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).reshape(-1)
    sd = student_feat.unsqueeze(0) - student_feat.unsqueeze(1)
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).reshape(-1)
    loss_a = F.smooth_l1_loss(s_angle, t_angle)
    return loss_d, loss_a


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=0):
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


def infer_feature_shapes(teacher, student, image_size, device):
    x = torch.randn(2, 3, image_size, image_size, device=device)
    with torch.no_grad():
        t_feat = forward_features(teacher, x)
        s_feat = forward_features(student, x)
    return int(t_feat.shape[1]), int(s_feat.shape[1])


def train_one_method(args, method: str, teacher, train_loader, val_loader, num_classes: int, device: torch.device):
    out_dir = args.output_dir / method / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[start] method={method} student={STUDENT_SPEC.model_name} seed={args.seed}", flush=True)
    student = create_student(args.weights_dir, num_classes, pretrained=not args.no_pretrained).to(device)
    adapter = None
    if method == "fitnet":
        teacher_ch, student_ch = infer_feature_shapes(teacher, student, args.image_size, device)
        adapter = nn.Conv2d(student_ch, teacher_ch, kernel_size=1).to(device)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    params = list(student.parameters()) + ([] if adapter is None else list(adapter.parameters()))
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = GradScaler(enabled=device.type == "cuda")
    best = {"top1": -math.inf, "epoch": 0}
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        student.train()
        if adapter is not None:
            adapter.train()
        total_loss = 0.0
        total = 0
        iterator = tqdm(train_loader, desc=f"{method} e{epoch}", leave=False, disable=not sys.stderr.isatty())
        for step, (images, targets) in enumerate(iterator, start=1):
            if args.max_train_batches and step > args.max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                student_logits = student(images)
                ce = ce_criterion(student_logits, targets)
                loss = ce
                loss_parts = {"ce": float(ce.detach().cpu())}
                if method != "ce":
                    with torch.no_grad():
                        teacher_logits = teacher(images)
                    if method == "kd":
                        loss_kd = kd_loss(student_logits, teacher_logits, args.temperature)
                        loss = (1.0 - args.kd_alpha) * ce + args.kd_alpha * loss_kd
                        loss_parts["kd"] = float(loss_kd.detach().cpu())
                    elif method == "dkd":
                        loss_dkd = dkd_loss(student_logits, teacher_logits, targets, args.dkd_alpha, args.dkd_beta, args.temperature)
                        loss = ce + loss_dkd
                        loss_parts["dkd"] = float(loss_dkd.detach().cpu())
                    elif method == "fitnet":
                        with torch.no_grad():
                            teacher_feat = forward_features(teacher, images)
                        student_feat = forward_features(student, images)
                        loss_hint = fitnet_loss(student_feat, teacher_feat, adapter)
                        loss = ce + args.hint_weight * loss_hint
                        loss_parts["hint"] = float(loss_hint.detach().cpu())
                    elif method == "rkd":
                        with torch.no_grad():
                            teacher_feat = pooled_features(teacher, images)
                        student_feat = pooled_features(student, images)
                        loss_d, loss_a = rkd_loss(student_feat, teacher_feat)
                        loss = ce + args.rkd_distance_weight * loss_d + args.rkd_angle_weight * loss_a
                        loss_parts["rkd_distance"] = float(loss_d.detach().cpu())
                        loss_parts["rkd_angle"] = float(loss_a.detach().cpu())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss method={method} epoch={epoch} step={step}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
        val_metrics = evaluate(student, val_loader, ce_criterion, device, args.max_val_batches)
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
            ckpt = {
                "method": method,
                "student_spec": STUDENT_SPEC.__dict__,
                "teacher_spec": TEACHER_SPEC.__dict__,
                "classes": train_loader.dataset.classes,
                "epoch": epoch,
                "state_dict": student.state_dict(),
                "adapter_state_dict": None if adapter is None else adapter.state_dict(),
                "metrics": best,
                "args": json_safe(vars(args)),
                "env": env_snapshot(),
            }
            torch.save(ckpt, out_dir / "best.pt")

    elapsed = time.time() - started
    summary = {
        "method": method,
        "student": STUDENT_SPEC.display_name,
        "student_model": STUDENT_SPEC.model_name,
        "teacher": TEACHER_SPEC.display_name,
        "teacher_model": TEACHER_SPEC.model_name,
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
    print(f"[done] method={method}: top1={best['top1']:.2f} top5={best['top5']:.2f}", flush=True)


def record_failure(args, method: str, exc: Exception) -> None:
    out_dir = args.output_dir / method / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": method,
        "student": STUDENT_SPEC.display_name,
        "student_model": STUDENT_SPEC.model_name,
        "teacher": TEACHER_SPEC.display_name,
        "teacher_model": TEACHER_SPEC.model_name,
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
    print(f"[failed] method={method}: {exc!r}", flush=True)


@torch.no_grad()
def eval_teacher(args, teacher, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(teacher, val_loader, criterion, device, args.max_val_batches)
    row = {
        "method": "teacher",
        "student": TEACHER_SPEC.display_name,
        "student_model": TEACHER_SPEC.model_name,
        "teacher": "",
        "teacher_model": "",
        "seed": args.seed,
        "epochs": 0,
        "best_epoch": 0,
        "best_top1": metrics["top1"],
        "best_top5": metrics["top5"],
        "best_loss": metrics["loss"],
        "elapsed_sec": 0,
        "checkpoint": str(args.teacher_ckpt),
        "status": "ok",
        "error": "",
    }
    (args.output_dir / "teacher_eval.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    append_csv(args.output_dir / "results.csv", row)
    print(f"[teacher] top1={metrics['top1']:.2f} top5={metrics['top5']:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/kd_imagewoof"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--teacher-ckpt", type=Path, default=Path("weights/teacher_mobilenetv3_large_best.pt"))
    parser.add_argument("--methods", default="ce,kd,dkd,fitnet,rkd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument("--dkd-alpha", type=float, default=1.0)
    parser.add_argument("--dkd-beta", type=float, default=8.0)
    parser.add_argument("--hint-weight", type=float, default=5.0)
    parser.add_argument("--rkd-distance-weight", type=float, default=25.0)
    parser.add_argument("--rkd-angle-weight", type=float, default=50.0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train_loader, val_loader, classes = build_loaders(args.data_root, args.image_size, args.batch_size, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(args.weights_dir, args.teacher_ckpt, len(classes), device)
    manifest = {
        "env": env_snapshot(),
        "data_root": str(args.data_root),
        "classes": classes,
        "num_train": len(train_loader.dataset),
        "num_val": len(val_loader.dataset),
        "args": json_safe(vars(args)),
        "teacher_spec": TEACHER_SPEC.__dict__,
        "student_spec": STUDENT_SPEC.__dict__,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    eval_teacher(args, teacher, val_loader, device)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    missing = [m for m in methods if m not in METHODS]
    if missing:
        raise ValueError(f"Unknown methods {missing}; known={METHODS}")
    for method in methods:
        try:
            train_one_method(args, method, teacher, train_loader, val_loader, len(classes), device)
        except Exception as exc:
            record_failure(args, method, exc)


if __name__ == "__main__":
    main()
