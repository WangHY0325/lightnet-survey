#!/bin/bash
#SBATCH --job-name=csi_env
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/csi_env_%j.out

set -e
cd /gpool/home/wanghongyang/WangHY/LightNet/Comm
export PATH=/gpool/home/wanghongyang/.conda/envs/LightNet/bin:$PATH

echo "=== env ==="
python -c "import torch; print('torch', torch.__version__)"
python -c "import scipy; print('scipy', scipy.__version__)"
python -c "import thop; print('thop ok')"
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import pandas; print('pandas', pandas.__version__)"

echo "=== models (no data needed) ==="
python - <<'PY'
import torch
from src.models import build_model
from src.utils import count_total_params, count_encoder_complexity

print(f"{'model':9s} {'cr':>2s}  {'out_shape':>22s}  "
      f"{'total':>11s}  {'enc_p':>11s}  {'enc_flops':>13s}")
print('-' * 80)
for m in ("csinet", "crissnet"):
    for cr in (4, 16, 64):
        net = build_model(m, reduction=cr).eval()
        with torch.no_grad():
            x = torch.randn(2, 2, 32, 32)
            y = net(x)
        tp = count_total_params(net)
        ec = count_encoder_complexity(net)
        print(f"{m:9s} {cr:>2d}  {str(tuple(y.shape)):>22s}  "
              f"{tp['params_total']:>11,}  {ec['params_encoder']:>11,.0f}  "
              f"{ec['flops_encoder']:>13,.0f}")
PY

echo DONE_csi_env
