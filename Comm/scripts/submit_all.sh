#!/bin/bash
# Submit the 12-job CSI feedback grid + final collect step.
#
# Usage:
#   ./scripts/submit_all.sh
#
# Run from the Comm/ directory on the server, AFTER:
#   1. data/cost2100/ is fully populated with all 6 .mat files
#   2. (optional but recommended) sbatch scripts/csi_sanity.slurm  has finished green
#   3. (optional) sbatch scripts/csi_smoke.slurm  has finished green

set -e
cd "$(dirname "$0")/.."

mkdir -p logs results checkpoints

echo "=== Phase 1: 12 training jobs (independent, all parallel) ==="
JIDS=""
for tag in csinet_in_cr4 csinet_in_cr16 csinet_in_cr64 \
           csinet_out_cr4 csinet_out_cr16 csinet_out_cr64 \
           crissnet_in_cr4 crissnet_in_cr16 crissnet_in_cr64 \
           crissnet_out_cr4 crissnet_out_cr16 crissnet_out_cr64; do
    j=$(sbatch --parsable "scripts/csi_${tag}.slurm")
    echo "  ${tag}=${j}"
    JIDS="${JIDS}:${j}"
done

# Strip leading colon
JIDS="${JIDS#:}"

echo "=== Phase 2: collect (depends on all 12) ==="
JC=$(sbatch --parsable --dependency=afterok:${JIDS} scripts/csi_collect.slurm)
echo "  csi_collect=${JC}"

echo
echo "============================================"
echo "Submitted 12 training + 1 collect job."
echo "Watch with:    squeue -u \$USER"
echo "Result file:   results/csi_summary.csv  (after collect job ${JC} finishes)"
echo "============================================"
