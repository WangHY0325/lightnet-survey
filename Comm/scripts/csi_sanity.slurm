#!/bin/bash
#SBATCH --job-name=csi_sanity
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/csi_sanity_%j.out

set -e
cd /gpool/home/wanghongyang/WangHY/LightNet/Comm
export PATH=/gpool/home/wanghongyang/.conda/envs/LightNet/bin:$PATH

echo "=== env ==="
python -c "import torch; print('torch', torch.__version__)"
python -c "import scipy; print('scipy', scipy.__version__)"
python -c "import thop; print('thop ok')"
echo "=== data ==="
python -c "from src.dataset import load_split; t = load_split('data/cost2100', 'in', 'train'); print('train_in shape', tuple(t.shape), 'dtype', t.dtype)"
python -c "from src.dataset import load_split; t = load_split('data/cost2100', 'in', 'val'); print('val_in   shape', tuple(t.shape))"
python -c "from src.dataset import load_split; t = load_split('data/cost2100', 'in', 'test'); print('test_in  shape', tuple(t.shape))"
python -c "from src.dataset import load_split; t = load_split('data/cost2100', 'out', 'train'); print('train_out shape', tuple(t.shape))"
python -c "from src.dataset import load_split; t = load_split('data/cost2100', 'out', 'val'); print('val_out  shape', tuple(t.shape))"
python -c "from src.dataset import load_split; t = load_split('data/cost2100', 'out', 'test'); print('test_out shape', tuple(t.shape))"
echo "=== models ==="
python - <<'PY'
import torch
from src.models import build_model
from src.utils import count_total_params, count_encoder_complexity
for m in ("csinet", "crissnet"):
    for cr in (4, 16, 64):
        net = build_model(m, reduction=cr).eval()
        x = torch.randn(2, 2, 32, 32)
        y = net(x)
        tp = count_total_params(net)
        ec = count_encoder_complexity(net)
        print(f"{m:9s} cr={cr:>2d}  out={tuple(y.shape)}  "
              f"total={tp['params_total']:>9,}  "
              f"enc_p={ec['params_encoder']:>10,.0f}  "
              f"enc_flops={ec['flops_encoder']:>12,.0f}")
PY
echo DONE
