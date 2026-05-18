from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METHOD_ORDER = ["baseline", "svd075", "svd050", "svd025", "svd050_recovery"]


def fmt_num(value, decimals=2):
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/lowrank_imagewoof"))
    args = parser.parse_args()
    results_csv = args.results_dir / "results.csv"
    latency_csv = args.results_dir / "latency.csv"
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)
    if not latency_csv.exists():
        raise FileNotFoundError(latency_csv)
    results = pd.read_csv(results_csv)
    latency = pd.read_csv(latency_csv)
    merged = pd.merge(results, latency, on=["method"], how="outer", suffixes=("", "_lat"))
    merged["params_m"] = pd.to_numeric(merged["params"], errors="coerce") / 1e6
    merged["gmacs"] = pd.to_numeric(merged["macs"], errors="coerce") / 1e9
    merged["order"] = merged["method"].map({method: idx for idx, method in enumerate(METHOD_ORDER)}).fillna(999)
    merged = merged.sort_values("order").drop(columns=["order"])
    summary_path = args.results_dir / "summary.csv"
    merged.to_csv(summary_path, index=False)

    ok = merged[(merged.get("status", "ok") == "ok") & (merged.get("status_lat", "ok") == "ok")]
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Method & Rank & Top-1 & Top-5 & Params (M) & GMACs & Latency (ms) \\\\",
        "\\midrule",
    ]
    for _, row in ok.iterrows():
        lines.append(
            f"{row['label']} & "
            f"{row['rank_setting']} & "
            f"{fmt_num(row.get('top1'))} & "
            f"{fmt_num(row.get('top5'))} & "
            f"{fmt_num(row.get('params_m'))} & "
            f"{fmt_num(row.get('gmacs'))} & "
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
                "note": "Controlled ImageWoof2-160 pointwise-Conv SVD stress/recovery case study; not a full low-rank method reproduction.",
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
