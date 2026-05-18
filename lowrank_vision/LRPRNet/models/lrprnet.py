import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor=8, min_value=None):
    """
    确保所有层的通道数能被 divisor 整除。

    Args:
        v (float): 原始通道数
        divisor (int): 除数，通常为 8
        min_value (int): 最小值，如果未指定则使用 divisor

    Returns:
        int: 调整后的通道数
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保向下调整的幅度不超过10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LRPRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6, rank_ratio=0.125):
        super().__init__()
        self.stride = stride

        mid_channels = _make_divisible(in_channels * expansion)

        # === 修改点1: 先深度卷积 ===
        # 这是共享的深度卷积层
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        # 扩展层（在深度卷积之后）
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True)
        )

        # 低秩点积卷积（两个1x1卷积）
        self.rank = max(8, int(mid_channels * rank_ratio))
        self.low_rank_conv = nn.Sequential(
            nn.Conv2d(mid_channels, self.rank, 1, bias=False),
            nn.BatchNorm2d(self.rank),
            nn.ReLU6(inplace=True),
            nn.Conv2d(self.rank, in_channels, 1, bias=False),  # 输出通道回到in_channels
            nn.BatchNorm2d(in_channels)
        )

        # === 修改点2: 最后可能需要1x1卷积调整通道数 ===
        self.use_residual = (stride == 1 and in_channels == out_channels)
        if not self.use_residual:
            self.adjust = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
            self.bn_adjust = nn.BatchNorm2d(out_channels)
        else:
            self.adjust = nn.Identity()
            self.bn_adjust = nn.Identity()

    def forward(self, x):
        identity = x

        # 步骤1: 先进行深度卷积（共享权重）
        x_depth = self.depthwise(x)  # D ⊗ F_in

        # 步骤2: 主路径（扩展 + 低秩点积）
        x_main = self.expand(x_depth)
        x_main = self.low_rank_conv(x_main)

        # 步骤3: L2归一化
        x_main = F.normalize(x_main, p=2, dim=1)

        # 步骤4: 残差连接 (D ⊗ F_in + 主路径输出)
        # 注意：这里x_main的输出通道是in_channels
        x = x_depth + x_main  # (P(2)P(1) + I) ⊗ (D ⊗ F_in)

        # 步骤5: 如果需要，调整通道数
        x = self.adjust(x)
        x = self.bn_adjust(x)

        return x


class LRPRNet(nn.Module):
    """基于MobileNetV2架构的LRPRNet完整实现"""

    def __init__(self, num_classes=10, width_mult=1.0, rank_ratio=0.125):
        super().__init__()
        block = LRPRBlock
        input_channel = 32
        last_channel = 1280

        # 初始卷积层
        input_channel = _make_divisible(input_channel * width_mult)
        # 使用 _make_divisible 确保最终层通道数能被8整除
        self.last_channel = _make_divisible(last_channel * width_mult)  # 修改这里

        features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, t, rank_ratio))
                input_channel = output_channel

        # 最终层
        features.append(nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        ))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # GAP
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)