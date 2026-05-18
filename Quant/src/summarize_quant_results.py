from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METHOD_ORDER = [
    "fp32_cuda",
    "fp16_cuda",
    "int8_dynamic_cpu",
    "int8_static_per_tensor_cpu",
    "int8_static_per_channel_cpu",
]


def fmt_num(value, decimals=2):
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/quant_imagewoof"))
    args = parser.parse_args()
    results_csv = args.results_dir / "results.csv"
    latency_csv = args.results_dir / "latency.csv"
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)
    if not latency_csv.exists():
        raise FileNotFoundError(latency_csv)
    results = pd.read_csv(results_csv)
    latency = pd.read_csv(latency_csv)
    merged = pd.merge(results, latency, on=["method", "runtime"], how="outer", suffixes=("", "_lat"))
    merged["params_m"] = pd.to_numeric(merged["params"], errors="coerce") / 1e6
    merged["order"] = merged["method"].map({method: idx for idx, method in enumerate(METHOD_ORDER)}).fillna(999)
    merged = merged.sort_values("order").drop(columns=["order"])
    summary_path = args.results_dir / "summary.csv"
    merged.to_csv(summary_path, index=False)

    ok = merged[(merged.get("status", "ok") == "ok") & (merged.get("status_lat", "ok") == "ok")]
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Method & Runtime & Top-1 & Top-5 & Params (M) & Size (MB) & Latency (ms) \\\\",
        "\\midrule",
    ]
    for _, row in ok.iterrows():
        lines.append(
            f"{row['label']} & "
            f"{row['runtime']} & "
            f"{fmt_num(row.get('top1'))} & "
            f"{fmt_num(row.get('top5'))} & "
            f"{fmt_num(row.get('params_m'))} & "
            f"{fmt_num(row.get('size_mb'))} & "
            f"{fmt_num(row.get('latency_mean_ms'))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path = args.results_dir / "summary_table.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = args.results_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "results_csv": str(results_csv),
                "latency_csv": str(latency_csv),
                "summary_csv": str(summary_path),
                "summary_table_tex": str(tex_path),
                "note": "Controlled ImageWoof2-160 MobileNetV2 quantization case study. CUDA and CPU latency rows are runtime-specific and should not be interpreted as mobile-device INT8 latency.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
