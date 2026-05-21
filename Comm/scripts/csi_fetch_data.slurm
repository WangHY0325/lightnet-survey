#!/bin/bash
#SBATCH --job-name=csi_fetch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/csi_fetch_%j.out

# Try to fetch the pre-processed COST2100 dataset from the Wen & Jin
# Google Drive folder using gdown. The cluster's compute node may have
# direct internet access where the local PC's proxy did not.
#
# If this job fails, fall back to: (1) browser download on local PC,
# then scp the six .mat files to data/cost2100/.

set -e
cd /gpool/home/wanghongyang/WangHY/LightNet/Comm
export PATH=/gpool/home/wanghongyang/.conda/envs/LightNet/bin:$PATH

mkdir -p data/cost2100

# Make sure gdown is available in the LightNet env. Install on the fly
# if missing (writes to user's pip cache, harmless).
python -c "import gdown" 2>/dev/null || pip install --quiet --user gdown

echo "=== gdown --folder attempt ==="
python -m gdown --folder \
    "https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj" \
    -O data/cost2100 2>&1 | head -60 || true

echo
echo "=== final tree ==="
ls -lh data/cost2100/ || true

# Verify with the actual loader. If the dataset has been placed correctly,
# every (scenario, split) returns a (N, 2, 32, 32) tensor.
python - <<'PY'
import sys
ok = True
try:
    from src.dataset import load_split
except Exception as e:
    print("import failed:", e)
    sys.exit(1)

for sc in ("in", "out"):
    for split in ("train", "val", "test"):
        try:
            t = load_split("data/cost2100", sc, split)
            print(f"  OK {sc:3s} {split:5s} shape={tuple(t.shape)} dtype={t.dtype}")
        except Exception as e:
            print(f"  MISSING {sc:3s} {split:5s}: {type(e).__name__}: {e}")
            ok = False
print("=== ALL OK ===" if ok else "=== INCOMPLETE: please scp manually ===")
PY

echo DONE_csi_fetch
