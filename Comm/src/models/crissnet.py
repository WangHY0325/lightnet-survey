"""CRissNet (TSnet backbone) — Wang, Teng, Zhao, Yu, Lau, IEEE TCCN 2025.

This module is a clean port of the upstream reference implementation at
https://github.com/CRissNet/CRissNet (file ``models/TSnet.py``). Behaviour and
weight initialization match the upstream commit on 2026-05-20. The only
edits are:

 * replaced the upstream ``utils.logger.info`` call with ``logging``;
 * exposed ``encode``/``decode`` so the per-side FLOPs/params counter can
   profile the UE encoder in isolation (matching CLNet's reporting style);
 * dropped a stray ``# pragma`` and the upstream Chinese comments that did
   not survive UTF-8 round-trip cleanly. Logic is otherwise unchanged.

NOTICE: Upstream repository is public on github.com/CRissNet/CRissNet
without an explicit license file as of 2026-05-20. We treat the code as
"academic research release". Cited as wang2025crissnet in the LightNetV9
bibliography. See LICENSES/CRissNet_NOTICE.md for redistribution notes.
"""

from collections import OrderedDict
import logging

import torch
import torch.nn as nn
from torch.nn import Softmax


__all__ = ["CRissNet", "build_crissnet"]


_log = logging.getLogger(__name__)
_TOTAL = 2 * 32 * 32  # 2048
_C = 2


# ---------------------------------------------------------------------------
# building blocks (verbatim ports)
# ---------------------------------------------------------------------------


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super().__init__(OrderedDict([
            ("conv", nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ("bn", nn.BatchNorm2d(out_planes)),
        ]))


class CRBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ("conv3x3", ConvBN(2, 10, 3)),
            ("relu1", nn.PReLU(num_parameters=10, init=0.3)),
            ("conv1x9", ConvBN(10, 10, [1, 9])),
            ("relu2", nn.PReLU(num_parameters=10, init=0.3)),
            ("conv9x1", ConvBN(10, 10, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ("conv1x5", ConvBN(2, 10, [5, 1])),
            ("relu", nn.PReLU(num_parameters=10, init=0.3)),
            ("conv5x1", ConvBN(10, 10, [1, 5])),
        ]))
        self.conv1x1 = ConvBN(20, 2, 1)
        self.relu1 = nn.PReLU(num_parameters=20, init=0.3)
        self.relu2 = nn.PReLU(num_parameters=2, init=0.3)

    def forward(self, x):
        identity = x
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu1(out)
        out = self.conv1x1(out)
        return self.relu2(out + identity)


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention; verbatim port of Crissnet+ basic variant."""

    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.size()
        q = self.query_conv(x)
        q_h = q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        q_w = q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
        k = self.key_conv(x)
        k_h = k.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        k_w = k.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
        v = self.value_conv(x)
        v_h = v.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        v_w = v.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
        e_h = torch.bmm(q_h, k_h).view(b, w, h, h).permute(0, 2, 1, 3)
        e_w = torch.bmm(q_w, k_w).view(b, h, w, w)
        att = self.softmax(torch.cat([e_h, e_w], 3))
        a_h = att[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
        a_w = att[:, :, :, h:h + w].contiguous().view(b * h, w, w)
        out_h = torch.bmm(v_h, a_h.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        out_w = torch.bmm(v_w, a_w.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)
        return self.gamma * (out_h + out_w) + x


class _Depthwise(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, [5, 1], padding=[2, 0], groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, [1, 5], padding=[0, 2], groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
        )
        self.conv4 = ConvBN(in_dim, in_dim, 1)

    def forward(self, x):
        out = self.conv4(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


# ---------------------------------------------------------------------------
# main model
# ---------------------------------------------------------------------------


class _TSnet(nn.Module):
    """Backbone of CRissNet (Crissnet basic variant). Total = 2*32*32."""

    def __init__(self, reduction: int = 4) -> None:
        super().__init__()
        if _TOTAL % reduction != 0:
            raise ValueError(f"reduction {reduction} does not divide {_TOTAL}")
        _log.info(f"CRissNet TSnet reduction={reduction}")
        self.reduction = reduction
        self.codeword_dim = _TOTAL // reduction

        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(_C, 4, 3)),
            ("relu1", nn.PReLU(num_parameters=4, init=0.3)),
            ("conv1x9_bn", ConvBN(4, 4, [1, 9])),
            ("relu2", nn.PReLU(num_parameters=4, init=0.3)),
            ("conv9x1_bn", ConvBN(4, 4, [9, 1])),
            ("relu3", nn.PReLU(num_parameters=4, init=0.3)),
        ]))
        self.attention = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(4, 48, 1)),
            ("relu1", nn.PReLU(num_parameters=48, init=0.3)),
            ("Criss-Cross", CrissCrossAttention(48)),
        ]))
        self.down = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(48, _C, 1)),
            ("relu1", nn.PReLU(num_parameters=_C, init=0.3)),
        ]))
        self.encoder2 = nn.Sequential(OrderedDict([
            ("DW", _Depthwise(_C)),
            ("relu", nn.PReLU(num_parameters=_C, init=0.3)),
        ]))
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.PReLU(num_parameters=4, init=0.3)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.PReLU(num_parameters=2, init=0.3)),
        ]))
        # convdown: stride (1,2) along width to halve W from 32 -> 16, so flat = 2*32*16 = 1024.
        self.convdown = nn.Conv2d(2, 2, 5, (1, 2), padding=2)
        self.encoder_fc = nn.Linear(_TOTAL // 2, self.codeword_dim)

        self.decoder_fc = nn.Linear(self.codeword_dim, _TOTAL)
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.PReLU(num_parameters=2, init=0.3)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock()),
        ]))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in",
                                         nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        e1 = self.encoder1(x)
        e1 = self.attention(e1)
        e1 = self.down(e1)
        e1 = self.convdown(e1)
        e2 = self.encoder2(x)
        e2 = self.convdown(e2)
        out = torch.cat((e1, e2), dim=1)
        out = self.encoder_conv(out)
        return self.encoder_fc(out.view(n, -1))

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        n = code.size(0)
        z = self.decoder_fc(code).view(n, _C, 32, 32)
        z = self.decoder_feature(z)
        return self.sigmoid(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# Public alias: the upstream paper name is CRissNet but the module class
# inside their repo is TSnet. We export both, with a friendly factory.
class CRissNet(_TSnet):
    pass


def build_crissnet(reduction: int = 4) -> CRissNet:
    return CRissNet(reduction=reduction)
