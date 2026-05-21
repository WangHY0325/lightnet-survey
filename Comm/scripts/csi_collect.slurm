#!/bin/bash
#SBATCH --job-name=csi_collect
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/csi_collect_%j.out

set -e
cd /gpool/home/wanghongyang/WangHY/LightNet/Comm
export PATH=/gpool/home/wanghongyang/.conda/envs/LightNet/bin:$PATH

python -m src.collect
echo DONE_collect
