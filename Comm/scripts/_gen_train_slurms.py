"""Generate 12 training SLURM scripts for the (model x scenario x cr) grid.

Run locally before scp to populate scripts/csi_<model>_<sc>_cr<n>.slurm.
"""

from pathlib import Path

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=csi_{tag}
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/csi_{tag}_%j.out

set -e
cd /gpool/home/wanghongyang/WangHY/LightNet/Comm
export PATH=/gpool/home/wanghongyang/.conda/envs/LightNet/bin:$PATH

python -m src.train \\
    --model {model} --scenario {sc} --reduction {cr} \\
    --epochs 200 --batch-size 200 --workers 8 \\
    --warmup-epochs 30 --val-every 2
echo DONE_{tag}
"""

OUT = Path(__file__).parent
for model in ("csinet", "crissnet"):
    for sc in ("in", "out"):
        for cr in (4, 16, 64):
            tag = f"{model}_{sc}_cr{cr}"
            (OUT / f"csi_{tag}.slurm").write_text(
                TEMPLATE.format(tag=tag, model=model, sc=sc, cr=cr),
                encoding="utf-8",
            )
            print(f"wrote scripts/csi_{tag}.slurm")
