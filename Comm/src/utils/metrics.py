"""Evaluation metric for CSI feedback.

NMSE definition follows the upstream CRissNet/CLNet release:
the network input is in [0, 1] (pre-normalized), so we de-centralize
by 0.5 before measuring the per-sample NMSE in dB.
"""

import torch


__all__ = ["nmse_db", "AverageMeter"]


class AverageMeter:
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n


def nmse_db(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-batch NMSE in dB.

    Returns a scalar tensor: 10 * log10( mean over batch of |H - H_hat|^2 / |H|^2 ).
    """
    with torch.no_grad():
        pred = pred - 0.5
        gt = gt - 0.5
        power_gt = gt[:, 0, :, :] ** 2 + gt[:, 1, :, :] ** 2
        diff = pred - gt
        mse = diff[:, 0, :, :] ** 2 + diff[:, 1, :, :] ** 2
        ratio = mse.sum(dim=(1, 2)) / power_gt.sum(dim=(1, 2)).clamp_min(1e-12)
        # Mean ratio first, then dB. Matches upstream evaluator.
        return 10.0 * torch.log10(ratio.mean())
