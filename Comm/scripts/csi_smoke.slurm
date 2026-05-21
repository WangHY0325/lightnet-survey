#!/bin/bash
#SBATCH --job-name=csi_smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/csi_smoke_%j.out

set -e
cd /gpool/home/wanghongyang/WangHY/LightNet/Comm
export PATH=/gpool/home/wanghongyang/.conda/envs/LightNet/bin:$PATH

# Smoke test: 5 epochs of csinet indoor cr=4, just to make sure the pipeline
# end-to-end works (data load -> train -> val NMSE -> ckpt save -> test).
python -m src.train \
    --model csinet --scenario in --reduction 4 \
    --epochs 5 --batch-size 200 --workers 4 \
    --warmup-epochs 1 --val-every 1 \
    --ckpt-dir checkpoints_smoke --results-dir results_smoke
echo SMOKE_DONE
