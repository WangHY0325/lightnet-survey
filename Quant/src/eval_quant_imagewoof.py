from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

from quant_common import (
    build_loaders,
    count_params,
    env_snapshot,
    evaluate,
    load_checkpoint,
    make_mobilenetv2,
    measure_latency,
    model_size_mb,
    parse_methods,
    set_seed,
    write_json,
    write_rows,
)


METHODS = (
    "fp32_cuda",
    "fp16_cuda",
    "int8_dynamic_cpu",
    "int8_static_per_tensor_cpu",
    "int8_static_per_channel_cpu",
)

METHOD_LABELS = {
    "fp32_cuda": ("FP32 CUDA baseline", "A30 CUDA"),
    "fp16_cuda": ("FP16 CUDA inference", "A30 CUDA"),
    "int8_dynamic_cpu": ("INT8 dynamic Linear", "CPU dynamic"),
    "int8_static_per_tensor_cpu": ("INT8 static PTQ per-tensor", "CPU static PTQ"),
    "int8_static_per_channel_cpu": ("INT8 static PTQ per-channel", "CPU static PTQ"),
}


def select_quant_engine() -> str:
    supported = torch.backends.quantized.supported_engines
    if "fbgemm" in supported:
        torch.backends.quantized.engine = "fbgemm"
    elif "qnnpack" in supported:
        torch.backends.quantized.engine = "qnnpack"
    return torch.backends.quantized.engine


def make_static_quantized_model(float_model, calib_loader, input_size: int, calib_batches: int, per_channel: bool):
    from torch.ao.quantization import QConfig, QConfigMapping
    from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    activation = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    if per_channel:
        weight = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    else:
        weight = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    qconfig = QConfig(activation=activation, weight=weight)
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    model = copy.deepcopy(float_model).cpu().eval()
    example_inputs = (torch.randn(1, 3, input_size, input_size),)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs)
    with torch.no_grad():
        for step, (images, _) in enumerate(calib_loader, start=1):
            if calib_batches and step > calib_batches:
                break
            prepared(images.cpu())
    return convert_fx(prepared).eval()


def failed_row(method: str, exc: Exception, params: int):
    label, runtime = METHOD_LABELS[method]
    env = env_snapshot()
    result = {
        "method": method,
        "label": label,
        "runtime": runtime,
        "top1": "",
        "top5": "",
        "samples": "",
        "params": params,
        "size_mb": "",
        "status": "failed",
        "error": repr(exc),
        **{f"env_{k}": v for k, v in env.items()},
    }
    latency = {
        "method": method,
        "runtime": runtime,
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
    parser.add_argument("--output-dir", type=Path, default=Path("results/quant_imagewoof"))
    parser.add_argument("--methods", default="all")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-batches", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--latency-batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=500)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    select_quant_engine()
    _, calib_loader, val_loader, classes = build_loaders(
        args.data_root, args.image_size, args.batch_size, args.workers, train_shuffle=False
    )
    base_model = make_mobilenetv2(num_classes=len(classes), pretrained_path=None)
    load_checkpoint(base_model, args.checkpoint, map_location="cpu")
    base_params = count_params(base_model)

    results = []
    latency_rows = []
    for method in parse_methods(args.methods, METHODS):
        label, runtime = METHOD_LABELS[method]
        print(f"[start] quant method={method} runtime={runtime}", flush=True)
        try:
            if method == "fp32_cuda":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = copy.deepcopy(base_model).to(device).eval()
                dtype = torch.float32
            elif method == "fp16_cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available for FP16 CUDA inference")
                device = torch.device("cuda")
                model = copy.deepcopy(base_model).to(device).half().eval()
                dtype = torch.float16
            elif method == "int8_dynamic_cpu":
                device = torch.device("cpu")
                model = torch.ao.quantization.quantize_dynamic(
                    copy.deepcopy(base_model).cpu().eval(), {nn.Linear}, dtype=torch.qint8
                )
                dtype = torch.float32
            elif method == "int8_static_per_tensor_cpu":
                device = torch.device("cpu")
                model = make_static_quantized_model(
                    base_model, calib_loader, args.image_size, args.calib_batches, per_channel=False
                )
                dtype = torch.float32
            elif method == "int8_static_per_channel_cpu":
                device = torch.device("cpu")
                model = make_static_quantized_model(
                    base_model, calib_loader, args.image_size, args.calib_batches, per_channel=True
                )
                dtype = torch.float32
            else:
                raise ValueError(method)

            metrics = evaluate(model, val_loader, device=device, dtype=dtype, max_batches=args.max_val_batches)
            latency = measure_latency(
                model,
                device=device,
                input_size=args.image_size,
                batch_size=args.latency_batch_size,
                warmup=args.warmup,
                repeats=args.repeats,
                dtype=dtype,
            )
            env = env_snapshot()
            result_row = {
                "method": method,
                "label": label,
                "runtime": runtime,
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "samples": metrics["samples"],
                "params": base_params,
                "size_mb": model_size_mb(model),
                "status": "ok",
                "error": "",
                **{f"env_{k}": v for k, v in env.items()},
            }
            latency_row = {
                "method": method,
                "runtime": runtime,
                "batch_size": args.latency_batch_size,
                "input_size": args.image_size,
                **latency,
                "status": "ok",
                "error": "",
                **{f"env_{k}": v for k, v in env.items()},
            }
        except Exception as exc:
            print(f"[failed] quant method={method}: {exc!r}", flush=True)
            result_row, latency_row = failed_row(method, exc, base_params)
        results.append(result_row)
        latency_rows.append(latency_row)
        print(json.dumps(result_row, ensure_ascii=False), flush=True)
        print(json.dumps(latency_row, ensure_ascii=False), flush=True)

    write_rows(args.output_dir / "results.csv", results)
    write_rows(args.output_dir / "latency.csv", latency_rows)
    write_json(
        args.output_dir / "eval_manifest.json",
        {
            "checkpoint": args.checkpoint,
            "methods": parse_methods(args.methods, METHODS),
            "data_root": args.data_root,
            "calib_batches": args.calib_batches,
            "note": "Controlled ImageWoof2-160 MobileNetV2 quantization case study; INT8 rows use CPU runtime.",
            "env": env_snapshot(),
        },
    )


if __name__ == "__main__":
    main()
