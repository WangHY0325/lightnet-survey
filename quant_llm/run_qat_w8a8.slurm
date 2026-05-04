#!/bin/bash
#SBATCH --job-name=qat_w8a8_30m
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a30:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/qat_w8a8_%j.out
#SBATCH --error=logs/qat_w8a8_%j.err

set -euo pipefail
export CUDA_HOME=/opt/cuda/12.4
export PATH=/opt/cuda/12.4/bin:$PATH
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1

echo "Job Start: $(date)"
echo "Assigned Nodes: $SLURM_JOB_NODELIST"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"

# Change to QuEST directory (we'll use QuEST's code)
cd /gpool/home/wanghongyang/WangHY/QuEST
mkdir -p logs results/qat_w8a8

# Python from LightNet conda environment
PY="/gpool/home/wanghongyang/.conda/envs/LightNet/bin/python"

# Model configuration (30M - same as FP32 baseline)
export N_LAYER=6
export N_EMBD=640
export N_HEAD=5
export LR=0.0012
export TOKENS=3000000000
export BATCH_SIZE=8
export ACC_STEPS=64
export SEQUENCE_LENGTH=2048

# Quantization settings - W8A8 using LSQ
export W_QUANT="LSQPlusWeightQuantizer"
export A_QUANT="LSQPlusActivationQuantizer"
export BITS=8

echo "=== W8A8 QAT Training Configuration ==="
echo "Model: 30M (${N_LAYER} layers, ${N_EMBD} embd, ${N_HEAD} heads)"
echo "Weight Quantizer: ${W_QUANT} (${BITS}-bit)"
echo "Activation Quantizer: ${A_QUANT} (${BITS}-bit)"
echo "Learning Rate: ${LR}"
echo "Batch Size: ${BATCH_SIZE} x ${ACC_STEPS} = $((BATCH_SIZE * ACC_STEPS))"

# Run training
$PY -u src/main.py \
    --model llama \
    --n-layer ${N_LAYER} \
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --multiple-of 256 \
    --batch-size ${BATCH_SIZE} \
    --sequence-length ${SEQUENCE_LENGTH} \
    --acc-steps ${ACC_STEPS} \
    --lr ${LR} \
    --weight-decay 0.1 \
    --iterations 2861 \
    --warmup-steps 286 \
    --eval-interval 200 \
    --eval-batches 100 \
    --seed 42 \
    --data-seed 1337 \
    --dataset fineweb \
    --w-quant ${W_QUANT} \
    --a-quant ${A_QUANT} \
    --w-quant-kwargs '{"bits": '${BITS}'}' \
    --a-quant-kwargs '{"bits": '${BITS}'}' \
    --results-base-folder results/qat_w8a8

echo "=== Training Completed ==="
echo "Job End: $(date)"

# Extract final validation loss
echo "=== Final Validation Loss ==="
grep "Eval.*val_loss" logs/qat_w8a8_${SLURM_JOB_ID}.out | tail -5
