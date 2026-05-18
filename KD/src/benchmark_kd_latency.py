from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from thop import profile

from model_utils import STUDENT_SPEC, TEACHER_SPEC, create_student, create_timm_model, env_snapshot


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def measure(model, x, warmup: int, repeats: int, device):
    model.eval()
    with torch.no_grad():
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


def load_student_for_method(method: str, args, device):
    if method == "teacher":
        model = create_timm_model(TEACHER_SPEC, args.num_classes, args.weights_dir, pretrained=True)
        ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        return model.to(device), str(args.teacher_ckpt), TEACHER_SPEC.display_name, TEACHER_SPEC.model_name
    model = create_student(args.weights_dir, args.num_classes, pretrained=not args.no_pretrained)
    ckpt_path = args.results_dir / method / f"seed{args.seed}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for method={method}: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    loaded = str(ckpt_path)
    return model.to(device), loaded, STUDENT_SPEC.display_name, STUDENT_SPEC.model_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/kd_imagewoof"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--teacher-ckpt", type=Path, default=Path("weights/teacher_mobilenetv3_large_best.pt"))
    parser.add_argument("--methods", default="teacher,ce,kd,dkd,fitnet,rkd")
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
    for method in [m.strip().lower() for m in args.methods.split(",") if m.strip()]:
        try:
            model, loaded, display_name, model_name = load_student_for_method(method, args, device)
            x = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)
            macs = ""
            try:
                macs_value, _ = profile(model, inputs=(x,), verbose=False)
                macs = macs_value
            except Exception as exc:
                print(f"[warn] thop failed for {method}: {exc}", flush=True)
            row = {
                "method": method,
                "display_name": display_name,
                "model_name": model_name,
                "batch_size": args.batch_size,
                "input_size": args.input_size,
                "params": count_params(model),
                "macs": macs,
                "checkpoint_loaded": loaded,
                "status": "ok",
                "error": "",
                **measure(model, x, args.warmup, args.repeats, device),
                **{f"env_{k}": v for k, v in env_snapshot().items()},
            }
        except Exception as exc:
            row = {
                "method": method,
                "display_name": "",
                "model_name": "",
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
            print(f"[failed] latency {method}: {exc!r}", flush=True)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    append_rows(args.results_dir / "latency.csv", rows)


if __name__ == "__main__":
    main()
