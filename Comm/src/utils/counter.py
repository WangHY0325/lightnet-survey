"""Parameter and FLOPs counters.

We report total parameters via ``sum(p.numel())`` and encoder-side complexity
(UE-side cost) via the ``thop`` profiler. Encoder is what the user equipment
actually has to run before feedback, so it is the right cost to charge.
"""

from typing import Dict

import torch
import torch.nn as nn


__all__ = ["count_total_params", "count_encoder_complexity"]


def count_total_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": int(total), "params_trainable": int(train)}


def count_encoder_complexity(model: nn.Module, input_shape=(1, 2, 32, 32)) -> Dict[str, float]:
    """Profile only the encoder side (UE-side cost).

    Both CsiNet and CRissNet expose ``encode``. We synthesize a thin wrapper
    so ``thop`` can profile it like a normal model.
    """
    try:
        from thop import profile
    except ImportError as e:
        raise RuntimeError(
            "thop is required for encoder complexity counting "
            "(pip install thop)"
        ) from e

    class _EncoderWrap(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m.encode(x)

    wrap = _EncoderWrap(model)
    x = torch.randn(*input_shape)
    flops, params = profile(wrap, inputs=(x,), verbose=False)
    return {
        "flops_encoder": float(flops),
        "params_encoder": float(params),
    }
