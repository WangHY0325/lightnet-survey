from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from thop import profile

from model_registry import create_model, env_snapshot, get_spec, parse_model_keys


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def maybe_load_checkpoint(model, ckpt_path: Path, device):
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        return str(ckpt_path)
    return ""


@torch.no_grad()
def measure(model, x, warmup: int, repeats: int, device):
    model.eval()
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


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--results-dir", type=Path, default=Path("results/mobile_nas_imagewoof"))
    parser.add_argument("--input-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for key in parse_model_keys(args.models):
        spec = get_spec(key)
        try:
            model = create_model(key, num_classes=args.num_classes, pretrained=not args.no_pretrained).to(device)
            ckpt_path = args.results_dir / key / f"seed{args.seed}" / "best.pt"
            loaded = maybe_load_checkpoint(model, ckpt_path, device)
            x = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)
            macs = ""
            try:
                macs_value, _ = profile(model, inputs=(x,), verbose=False)
                macs = macs_value
            except Exception as exc:
                print(f"[warn] thop failed for {key}: {exc}", flush=True)
            metrics = measure(model, x, args.warmup, args.repeats, device)
            row = {
                "model_key": key,
                "display_name": spec.display_name,
                "source": spec.source,
                "model_name": spec.model_name,
                "paper_family": spec.paper_family,
                "batch_size": args.batch_size,
                "input_size": args.input_size,
                "params": count_params(model),
                "macs": macs,
                "checkpoint_loaded": loaded,
                "status": "ok",
                "error": "",
                **metrics,
                **{f"env_{k}": v for k, v in env_snapshot().items()},
            }
        except Exception as exc:
            row = {
                "model_key": key,
                "display_name": spec.display_name,
                "source": spec.source,
                "model_name": spec.model_name,
                "paper_family": spec.paper_family,
                "batch_size": args.batch_size,
                "input_size": args.input_size,
                "params": "",
                "macs": "",
                "checkpoint_loaded": "",
                "status": "failed",
                "error": repr(exc),
                "latency_mean_ms": "",
                "latency_p50_ms": "",
                "latency_p90_ms": "",
                "latency_min_ms": "",
                "latency_max_ms": "",
                **{f"env_{k}": v for k, v in env_snapshot().items()},
            }
            print(f"[failed] latency {spec.display_name}: {exc!r}", flush=True)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    append_rows(args.results_dir / "latency.csv", rows)


if __name__ == "__main__":
    main()
