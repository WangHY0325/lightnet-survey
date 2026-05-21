"""CSI feedback training entry point.

Trains either CsiNet or CRissNet on COST2100 indoor or outdoor split at one
compression ratio. Writes a JSON summary plus the best-NMSE checkpoint.

Usage example:
    python -m src.train --model crissnet --scenario in --reduction 4 \
        --data-dir data/cost2100 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from .dataset import Cost2100DataLoader
from .models import build_model
from .utils import (
    AverageMeter, WarmUpCosineAnnealingLR,
    count_encoder_complexity, count_total_params,
    nmse_db, seed_everything,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["csinet", "crissnet"])
    p.add_argument("--scenario", required=True, choices=["in", "out"])
    p.add_argument("--reduction", type=int, required=True,
                   choices=[4, 8, 16, 32, 64])
    p.add_argument("--data-dir", default="data/cost2100")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--lr-min", type=float, default=5e-5)
    p.add_argument("--warmup-epochs", type=int, default=30)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--print-freq", type=int, default=20,
                   help="batches between progress prints")
    p.add_argument("--val-every", type=int, default=2,
                   help="run val NMSE every K epochs")
    return p.parse_args()


def _run_id(args: argparse.Namespace) -> str:
    return f"csi_{args.model}_{args.scenario}_cr{args.reduction}_seed{args.seed}"


@torch.no_grad()
def evaluate(model: nn.Module, loader) -> tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter("loss")
    nmse_meter = AverageMeter("nmse")
    crit = nn.MSELoss(reduction="mean")
    for batch in loader:
        x = batch[0]
        y = model(x)
        loss = crit(y, x)
        n = x.size(0)
        loss_meter.update(loss.item(), n)
        nmse_meter.update(nmse_db(y, x).item(), n)
    return float(loss_meter.avg), float(nmse_meter.avg)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    run_id = _run_id(args)
    print(f"[{run_id}] args = {vars(args)}", flush=True)

    # Force CPU. The cluster's `cpu` partition has no GPU, but be defensive.
    torch.set_num_threads(max(1, args.workers))
    device = torch.device("cpu")

    loaders = Cost2100DataLoader(
        root=args.data_dir, scenario=args.scenario,
        batch_size=args.batch_size, num_workers=args.workers,
    )
    train_loader, val_loader, test_loader = loaders()
    print(f"[{run_id}] dataset sizes: "
          f"train={len(loaders.train)} val={len(loaders.val)} test={len(loaders.test)}",
          flush=True)

    model = build_model(args.model, reduction=args.reduction).to(device)
    total_p = count_total_params(model)
    enc_p = count_encoder_complexity(model)
    print(f"[{run_id}] total_params={total_p['params_total']:,} "
          f"encoder_params={enc_p['params_encoder']:,.0f} "
          f"encoder_flops={enc_p['flops_encoder']:,.0f}", flush=True)

    crit = nn.MSELoss(reduction="mean").to(device)
    optim = Adam(model.parameters(), lr=args.lr)
    iters_per_epoch = len(train_loader)
    sched = WarmUpCosineAnnealingLR(
        optim,
        T_max=args.epochs * iters_per_epoch,
        T_warmup=args.warmup_epochs * iters_per_epoch,
        eta_min=args.lr_min,
    )

    ckpt_dir = Path(args.ckpt_dir) / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{args.scenario}_cr{args.reduction}.pth"

    best_nmse_val = float("inf")
    best_epoch = -1
    history = []
    t_start = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter("train_loss")
        for batch_idx, batch in enumerate(train_loader):
            x = batch[0]
            y = model(x)
            loss = crit(y, x)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            loss_meter.update(loss.item(), x.size(0))
            if (batch_idx + 1) % args.print_freq == 0:
                lr_now = sched.get_lr()[0]
                print(f"[{run_id}] ep {ep:03d}/{args.epochs} "
                      f"iter {batch_idx+1}/{iters_per_epoch} "
                      f"lr={lr_now:.2e} loss={loss.item():.4e}", flush=True)

        record = {"epoch": ep, "train_loss": float(loss_meter.avg)}

        if ep % args.val_every == 0 or ep == args.epochs:
            v_loss, v_nmse = evaluate(model, val_loader)
            record.update({"val_loss": v_loss, "val_nmse_db": v_nmse})
            print(f"[{run_id}] ep {ep:03d} VAL  loss={v_loss:.4e} nmse={v_nmse:.3f} dB",
                  flush=True)
            if v_nmse < best_nmse_val:
                best_nmse_val = v_nmse
                best_epoch = ep
                torch.save({
                    "state_dict": model.state_dict(),
                    "epoch": ep,
                    "val_nmse_db": v_nmse,
                    "args": vars(args),
                }, str(best_path))
        history.append(record)

    elapsed = time.time() - t_start

    # Final test on the best checkpoint
    state = torch.load(str(best_path), map_location="cpu")
    model.load_state_dict(state["state_dict"])
    test_loss, test_nmse = evaluate(model, test_loader)
    print(f"[{run_id}] DONE best_epoch={best_epoch} "
          f"val_nmse={best_nmse_val:.3f} test_nmse={test_nmse:.3f} "
          f"elapsed={elapsed/60:.1f} min", flush=True)

    # Encoder size in MB (state_dict bytes for the encoder side only)
    enc_state_bytes = 0
    enc_module_prefixes = ("encoder", "attention", "down", "encoder_conv",
                           "convdown", "encoder1", "encoder2", "encoder_fc")
    for k, v in model.state_dict().items():
        if any(k.startswith(p) for p in enc_module_prefixes) and isinstance(v, torch.Tensor):
            enc_state_bytes += v.numel() * v.element_size()
    enc_size_mb = enc_state_bytes / (1024 * 1024)

    summary = {
        "run_id": run_id,
        "model": args.model,
        "scenario": args.scenario,
        "reduction": args.reduction,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_nmse_db": best_nmse_val,
        "test_nmse_db": test_nmse,
        "test_loss": test_loss,
        "elapsed_seconds": elapsed,
        "params_total": total_p["params_total"],
        "params_encoder": enc_p["params_encoder"],
        "flops_encoder": enc_p["flops_encoder"],
        "encoder_size_mb": enc_size_mb,
        "best_checkpoint": str(best_path),
        "history": history,
        "args": vars(args),
    }
    out = Path(args.results_dir) / f"{run_id}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[{run_id}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
