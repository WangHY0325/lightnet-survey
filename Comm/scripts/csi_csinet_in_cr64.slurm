#!/bin/bash
#SBATCH --job-name=csi_csinet_in_cr64
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/csi_csinet_in_cr64_%j.out

set -e
cd /gpool/home/wanghongyang/WangHY/LightNet/Comm
export PATH=/gpool/home/wanghongyang/.conda/envs/LightNet/bin:$PATH

python -m src.train \
    --model csinet --scenario in --reduction 64 \
    --epochs 200 --batch-size 200 --workers 8 \
    --warmup-epochs 30 --val-every 2
echo DONE_csinet_in_cr64
