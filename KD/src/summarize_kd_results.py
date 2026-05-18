from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METHOD_LABELS = {
    "teacher": "Teacher MobileNetV3-Large",
    "ce": "CE student",
    "kd": "Hinton KD",
    "dkd": "DKD",
    "fitnet": "FitNet",
    "rkd": "RKD",
}


def fmt_num(value, decimals=2):
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/kd_imagewoof"))
    args = parser.parse_args()
    results_csv = args.results_dir / "results.csv"
    latency_csv = args.results_dir / "latency.csv"
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)
    if not latency_csv.exists():
        raise FileNotFoundError(latency_csv)
    results = pd.read_csv(results_csv)
    latency = pd.read_csv(latency_csv)
    merged = pd.merge(results, latency, on=["method"], how="outer", suffixes=("_train", "_lat"))
    merged["gmacs"] = pd.to_numeric(merged["macs"], errors="coerce") / 1e9
    merged["params_m"] = pd.to_numeric(merged["params"], errors="coerce") / 1e6
    merged["label"] = merged["method"].map(METHOD_LABELS).fillna(merged["method"])
    summary_path = args.results_dir / "summary.csv"
    merged.to_csv(summary_path, index=False)

    ok = merged[(merged.get("status_train", "ok") == "ok") & (merged.get("status_lat", "ok") == "ok")]
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Top-1 & Top-5 & Params (M) & GMACs & Latency (ms) \\\\",
        "\\midrule",
    ]
    for _, row in ok.iterrows():
        lines.append(
            f"{row['label']} & "
            f"{fmt_num(row.get('best_top1'))} & "
            f"{fmt_num(row.get('best_top5'))} & "
            f"{fmt_num(row.get('params_m'))} & "
            f"{fmt_num(row.get('gmacs'))} & "
            f"{fmt_num(row.get('latency_mean_ms'))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path = args.results_dir / "summary_table.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = args.results_dir / "summary_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "results_csv": str(results_csv),
                "latency_csv": str(latency_csv),
                "summary_csv": str(summary_path),
                "summary_table_tex": str(tex_path),
                "note": "Controlled ImageWoof2-160 KD fine-tuning and server latency; not a full KD benchmark.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
