"""Aggregate per-run JSON summaries into a single CSV for the paper table.

Looks at every results/csi_*_summary.json and writes:
    results/csi_summary.csv
with columns:
    model, scenario, cr, nmse_db, params_total, params_encoder,
    flops_encoder, encoder_size_mb, train_seconds, best_epoch, run_id

Also emits a plain-text Markdown table to results/csi_summary_table.md.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path


_FIELDS = [
    "model", "scenario", "cr", "nmse_db", "params_total", "params_encoder",
    "flops_encoder", "encoder_size_mb", "train_seconds", "best_epoch", "run_id",
]


def _load_all(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("csi_*_summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        rows.append({
            "model": d.get("model", ""),
            "scenario": d.get("scenario", ""),
            "cr": d.get("reduction", ""),
            "nmse_db": round(float(d.get("test_nmse_db", float("nan"))), 3),
            "params_total": int(d.get("params_total", 0)),
            "params_encoder": int(float(d.get("params_encoder", 0))),
            "flops_encoder": int(float(d.get("flops_encoder", 0))),
            "encoder_size_mb": round(float(d.get("encoder_size_mb", 0.0)), 4),
            "train_seconds": round(float(d.get("elapsed_seconds", 0.0)), 1),
            "best_epoch": int(d.get("best_epoch", -1)),
            "run_id": d.get("run_id", path.stem),
        })
    return rows


def _emit_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _emit_md(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("(no rows)\n", encoding="utf-8")
        return
    header = "| " + " | ".join(_FIELDS) + " |"
    sep = "| " + " | ".join("---" for _ in _FIELDS) + " |"
    body = "\n".join(
        "| " + " | ".join(str(r[c]) for c in _FIELDS) + " |"
        for r in rows
    )
    path.write_text("\n".join([header, sep, body, ""]), encoding="utf-8")


def main() -> None:
    results_dir = Path("results")
    rows = _load_all(results_dir)
    if not rows:
        print("no per-run summaries found under results/", flush=True)
    # sort: model (csinet first), scenario (in first), cr ascending
    order_model = {"csinet": 0, "crissnet": 1}
    order_sc = {"in": 0, "out": 1}
    rows.sort(key=lambda r: (order_model.get(r["model"], 99),
                             order_sc.get(r["scenario"], 99),
                             int(r["cr"])))
    _emit_csv(rows, results_dir / "csi_summary.csv")
    _emit_md(rows, results_dir / "csi_summary_table.md")
    print(f"wrote results/csi_summary.csv with {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
