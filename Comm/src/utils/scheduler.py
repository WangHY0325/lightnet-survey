"""Cosine warm-up scheduler.

Direct port of the upstream CRissNet release at github.com/CRissNet/CRissNet
(file ``utils/scheduler.py``). Identical to the CRNet/CLNet upstream as well.
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


__all__ = ["WarmUpCosineAnnealingLR"]


class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0.0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / max(self.T_warmup, 1)
                    for base_lr in self.base_lrs]
        k = 1 + math.cos(math.pi *
                         (self.last_epoch - self.T_warmup) /
                         max(self.T_max - self.T_warmup, 1))
        return [self.eta_min + (base_lr - self.eta_min) * k / 2
                for base_lr in self.base_lrs]
