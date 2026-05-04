#!/bin/bash
#SBATCH --job-name=adalora
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs_adalora/adalora_%j.out
#SBATCH --error=logs_adalora/adalora_%j.err

set -euo pipefail
export CUDA_HOME=/opt/cuda/12.4
export PATH=/opt/cuda/12.4/bin:$PATH
export PYTHONUNBUFFERED=1

echo "Job Start: $(date)"
echo "Assigned Nodes: $SLURM_JOB_NODELIST"

PY="/gpool/home/wanghongyang/.conda/envs/LightNet/bin/python"
WORKDIR="/gpool/home/wanghongyang/WangHY/LightNet/Quant/LoRA"

cd "$WORKDIR"
mkdir -p logs_adalora

$PY -u train_adalora.py

echo "Job End: $(date)"
