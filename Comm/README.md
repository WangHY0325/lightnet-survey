# Communication-domain controlled reproductions for LightNetV9

Two CSI-feedback networks reproduced on COST2100 indoor + outdoor splits at three compression ratios. All training runs on the cluster's `cpu` partition under the `LightNet` conda env. Single seed (42), no variance reporting (consistent with the rest of the LightNetV9 controlled-reproduction series).

## Models

| Model       | Source                                              | Role                |
|-------------|-----------------------------------------------------|---------------------|
| `csinet`    | Wen, Shih, Jin, *IEEE WCL 2018* (CsiNet)            | classical baseline  |
| `crissnet`  | Wang, Teng et al., *IEEE TCCN 2025* (CRissNet+)     | 2025 lightweight    |

CsiNet is re-implemented in PyTorch from the architecture description in the original paper (FC bottleneck encoder + RefineNet residual decoder, 2x32x32 input, sigmoid output). CRissNet's TSnet is a faithful port of the upstream code at https://github.com/CRissNet/CRissNet (Apache-style permissive license, see `LICENSES/CRissNet_NOTICE.md`).

## Experiment grid

| Scenario | CR (1/n1) | n_jobs |
|----------|-----------|--------|
| indoor   | 4, 16, 64 | 3      |
| outdoor  | 4, 16, 64 | 3      |

Total: 2 models x 2 scenarios x 3 CR = **12 training runs** + 1 smoke test.

## Dataset

COST2100 pre-processed by Wen and Jin (the same dataset used by CsiNet, CRNet, CLNet, CRissNet). Layout expected:

```
data/cost2100/
├── DATA_Htrainin.mat      # ~10 MB
├── DATA_Hvalin.mat        # ~3 MB
├── DATA_Htestin.mat       # ~3 MB
├── DATA_Htrainout.mat     # ~10 MB
├── DATA_Hvalout.mat       # ~3 MB
└── DATA_Htestout.mat      # ~3 MB
```

Each `.mat` contains a single key `HT` with shape `(N, 2*32*32)`. The dataset loader reshapes it back to `(N, 2, 32, 32)`.

Download (manual, browser):
1. Open https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj
2. Download all six `.mat` files into `code/Comm/data/cost2100/`
3. `scp` the folder to the server `/gpool/home/wanghongyang/WangHY/LightNet/Comm/data/cost2100/`

A backup mirror via Baidu Netdisk is referenced by CLNet's README.

## Layout

```
Comm/
├── README.md
├── requirements.txt
├── data/
│   └── cost2100/        # six .mat files placed here
├── src/
│   ├── dataset/cost2100.py
│   ├── models/csinet.py
│   ├── models/crissnet.py
│   ├── utils/seed.py / scheduler.py / metrics.py / counter.py
│   ├── train.py
│   └── eval.py
├── scripts/
│   ├── csi_smoke.slurm
│   ├── csi_csinet_<sc>_<cr>.slurm    (6 jobs)
│   ├── csi_crissnet_<sc>_<cr>.slurm  (6 jobs)
│   ├── csi_collect.slurm
│   └── submit_all.sh
├── checkpoints/
├── results/
└── logs/
```

## Pipeline

```bash
# 0. one-time, on server
cd /gpool/home/wanghongyang/WangHY/LightNet/Comm
chmod +x scripts/submit_all.sh

# 1. smoke test (5 min, indoor cr=4, csinet, 2 epochs)
sbatch scripts/csi_smoke.slurm

# 2. full grid (12 jobs, all parallel after smoke clears)
./scripts/submit_all.sh

# 3. when all 12 finish: aggregate + emit table
sbatch scripts/csi_collect.slurm
```

## Output

`results/csi_summary.csv` with columns:
`model, scenario, cr, nmse_db, params_total, params_encoder, flops_encoder, encoder_size_mb, train_seconds, best_epoch`.
This file is the source of truth for the new `tab:comm_csi_repro` row block in `LightNetV9.tex`.

## Honesty contract

- Single seed (seed=42).
- 200 training epochs (CPU budget; original CRissNet paper uses 1500 GPU epochs). NMSE values may sit above the literature numbers, which is the intended observation: it is a controlled mechanism check under a CPU-bounded protocol, not a SOTA chase.
- All NMSE numbers come from the test split, evaluated with the same evaluator function as the upstream CRissNet repo (`utils/metrics.py::nmse_db`), to keep our value comparable to the upstream value.
- Encoder-only complexity is reported (UE-side cost). Decoder runs at the BS and is not bandwidth-limited.
- No hyperparameter search per (model, scenario, cr) cell. Optimizer Adam, lr cosine warm-up to 3e-3, batch=200.
