# models/model_loader.py
"""
由于你的MobileNetV2和LRPRNet代码可能在其他地方定义，
这里提供一个加载接口
"""
import sys
import os

# 添加原始代码路径
sys.path.append('/gpool/home/wanghongyang/WangHY/LRPRNet/LRPR-MobileNetV2')


def load_model_definitions():
    """加载模型定义"""
    try:
        # 尝试导入你的原始代码
        from mobilenetv2 import MobileNetV2
        from lrprnet import LRPRNet
        return MobileNetV2, LRPRNet
    except ImportError:
        print("Warning: Could not import original model definitions")
        print("Using simplified versions for visualization")

        # 如果导入失败，创建简化的版本
        import torch.nn as nn

        class SimpleMobileNetV2(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                # 简化的版本，仅用于演示
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU6(inplace=True),
                )
                self.classifier = nn.Linear(32, num_classes)

            def forward(self, x):
                x = self.features(x)
                x = x.mean([2, 3])
                x = self.classifier(x)
                return x

        class SimpleLRPRNet(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU6(inplace=True),
                )
                self.classifier = nn.Linear(32, num_classes)

            def forward(self, x):
                x = self.features(x)
                x = x.mean([2, 3])
                x = self.classifier(x)
                return x

        return SimpleMobileNetV2, SimpleLRPRNet


# 导出模型
MobileNetV2, LRPRNet = load_model_definitions()