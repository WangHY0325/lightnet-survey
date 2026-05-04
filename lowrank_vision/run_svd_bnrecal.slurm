#!/bin/bash
#SBATCH --job-name=svd_bnrecal
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/svd_bnrecal_%j.out
#SBATCH --error=logs/svd_bnrecal_%j.err

set -euo pipefail
export CUDA_HOME=/opt/cuda/12.4
export PATH=/opt/cuda/12.4/bin:$PATH
export PYTHONUNBUFFERED=1

echo "Job Start: $(date)"
echo "Assigned Nodes: $SLURM_JOB_NODELIST"

PY="/gpool/home/wanghongyang/.conda/envs/LightNet/bin/python"
DATA="/gpool/home/wanghongyang/WangHY/LightNet/data/imagewoof2-160"
CKPT="/gpool/home/wanghongyang/WangHY/LightNet/LowRank/weights/mobilenetv2_nas_v2_best.pt"
OUTDIR="/gpool/home/wanghongyang/WangHY/LightNet/LowRank/results/lowrank_imagewoof_recal"
SRCDIR="/gpool/home/wanghongyang/WangHY/LightNet/LowRank/src"

mkdir -p logs "$OUTDIR"

cd "$SRCDIR"
$PY -u run_lowrank_imagewoof.py \
    --data-root "$DATA" \
    --checkpoint "$CKPT" \
    --output-dir "$OUTDIR" \
    --methods "svd075_recal,svd050_recal,svd025_recal,svd050_recovery" \
    --image-size 160 \
    --batch-size 128 \
    --workers 6 \
    --seed 42 \
    --recovery-epochs 25 \
    --lr 3e-4 \
    --weight-decay 0.05 \
    --warmup-epochs 3 \
    --label-smoothing 0.1 \
    --latency-batch-size 1

echo "=== Results ==="
cat "$OUTDIR/results.csv"
echo "Job End: $(date)"
