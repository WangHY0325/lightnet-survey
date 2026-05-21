"""CsiNet — Wen, Shih, Jin (IEEE WCL 2018).

PyTorch re-implementation following the architecture described in the paper
("Deep learning for massive MIMO CSI feedback", arXiv:1712.08919). Cross-checked
against community reference: https://github.com/SeokhyunJeong/CsiNet-Pytorch and
https://github.com/SIJIEJI/CLNet (which is itself derived from the CRNet release).

Architecture summary
====================
Input:    H_a in R^{2 x 32 x 32} (real & imaginary stacked, pre-normalized to [0, 1]).
Encoder:  Conv2d(2 -> 2, k=3) -> BN -> LeakyReLU -> Flatten -> FC(2*32*32 -> M).
            Codeword length M = 2*32*32 // reduction.
Decoder:  FC(M -> 2*32*32) -> Reshape -> 2 x RefineNet block -> Sigmoid.

Each RefineNet block (residual) has three Conv2d-BN-LeakyReLU stages with
channel widths 8 -> 16 -> 2 and a skip connection back to the input of the block.
"""

from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn


__all__ = ["CsiNet", "build_csinet"]


_LEAKY = 0.3
_HW = 32
_C = 2
_TOTAL = _C * _HW * _HW


def _conv_bn(in_ch: int, out_ch: int, ksize: int = 3) -> nn.Sequential:
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_ch, out_ch, kernel_size=ksize,
                           stride=1, padding=ksize // 2, bias=False)),
        ("bn", nn.BatchNorm2d(out_ch)),
        ("act", nn.LeakyReLU(negative_slope=_LEAKY, inplace=True)),
    ]))


class _RefineNetBlock(nn.Module):
    """Residual decoder block as in CsiNet."""

    def __init__(self, channels: Sequence[int] = (8, 16, 2)) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.path = nn.Sequential(
            _conv_bn(_C, c1, 3),
            _conv_bn(c1, c2, 3),
            # Final stage of the block: BN but no activation (residual sum then
            # apply LeakyReLU outside; matches the original Keras code).
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c3),
        )
        self.act = nn.LeakyReLU(negative_slope=_LEAKY, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.path(x) + x)


class CsiNet(nn.Module):
    def __init__(self, reduction: int = 4) -> None:
        super().__init__()
        if _TOTAL % reduction != 0:
            raise ValueError(
                f"reduction={reduction} does not divide total feature dim {_TOTAL}"
            )
        self.reduction = reduction
        self.codeword_dim = _TOTAL // reduction

        self.encoder_conv = _conv_bn(_C, _C, ksize=3)
        self.encoder_fc = nn.Linear(_TOTAL, self.codeword_dim)

        self.decoder_fc = nn.Linear(self.codeword_dim, _TOTAL)
        self.refine = nn.Sequential(
            _RefineNetBlock(),
            _RefineNetBlock(),
        )
        self.head_conv = nn.Conv2d(_C, _C, kernel_size=3, stride=1, padding=1, bias=False)
        self.head_bn = nn.BatchNorm2d(_C)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=_LEAKY, mode="fan_in",
                                         nonlinearity="leaky_relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # The two encoder/decoder methods are exposed so the FLOPs/params counter
    # can profile the encoder in isolation (UE-side cost reporting).
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        z = self.encoder_conv(x)
        return self.encoder_fc(z.view(n, -1))

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        n = code.size(0)
        z = self.decoder_fc(code).view(n, _C, _HW, _HW)
        z = self.refine(z)
        z = self.head_bn(self.head_conv(z))
        return self.sigmoid(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def build_csinet(reduction: int = 4) -> CsiNet:
    return CsiNet(reduction=reduction)
