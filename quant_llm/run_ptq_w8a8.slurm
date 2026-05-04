#!/bin/bash
#SBATCH --job-name=ptq_w8a8_30m
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/ptq_w8a8_%j.out
#SBATCH --error=logs/ptq_w8a8_%j.err

set -euo pipefail
export CUDA_HOME=/opt/cuda/12.4
export PATH=/opt/cuda/12.4/bin:$PATH
export PYTHONUNBUFFERED=1

echo "Job Start: $(date)"
echo "Assigned Nodes: $SLURM_JOB_NODELIST"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"

# Change to LightNet Quant directory
cd /gpool/home/wanghongyang/WangHY/LightNet/Quant
mkdir -p logs results/ptq_w8a8

# Python from LightNet conda environment
PY="/gpool/home/wanghongyang/.conda/envs/LightNet/bin/python"

echo "=== W8A8 PTQ Evaluation ==="
echo "Model: 30M LLM"
echo "Method: Post-Training Quantization (Uniform W8A8)"

# Run PTQ evaluation
$PY -u eval_ptq_w8a8.py

echo "=== Evaluation Completed ==="
echo "Job End: $(date)"

# Show results
echo "=== Results Summary ==="
cat results/ptq_w8a8/results.json
