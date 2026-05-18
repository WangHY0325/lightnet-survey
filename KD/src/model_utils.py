from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import timm


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    model_name: str
    weight_file: str


TEACHER_SPEC = ModelSpec(
    "teacher_mobilenetv3_large",
    "Teacher MobileNetV3-Large",
    "mobilenetv3_large_100",
    "mobilenetv3_large_100_ra-f55367f5.pth",
)

STUDENT_SPEC = ModelSpec(
    "student_mobilenetv3_small_050",
    "Student MobileNetV3-Small-0.5",
    "mobilenetv3_small_050",
    "mobilenetv3_small_050_lambc-4b7bbe87.pth",
)


def env_snapshot() -> dict:
    info = {
        "torch": torch.__version__,
        "timm": timm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_runtime"] = torch.version.cuda
    return info


def create_timm_model(spec: ModelSpec, num_classes: int, weights_dir: Path, pretrained: bool = True) -> nn.Module:
    if not pretrained:
        return timm.create_model(spec.model_name, pretrained=False, num_classes=num_classes)
    weight_path = weights_dir / spec.weight_file
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Missing local pretrained weight file for {spec.model_name}: {weight_path}. "
            "Server-side model downloads are disabled for this experiment."
        )
    model = timm.create_model(spec.model_name, pretrained=False, num_classes=1000)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if isinstance(state, dict):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.reset_classifier(num_classes)
    return model


def load_teacher(weights_dir: Path, teacher_ckpt: Path, num_classes: int, device: torch.device) -> nn.Module:
    teacher = create_timm_model(TEACHER_SPEC, num_classes, weights_dir, pretrained=True)
    if teacher_ckpt.exists():
        ckpt = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        teacher.load_state_dict(state, strict=True)
    else:
        raise FileNotFoundError(f"Missing teacher checkpoint: {teacher_ckpt}")
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def create_student(weights_dir: Path, num_classes: int, pretrained: bool = True) -> nn.Module:
    return create_timm_model(STUDENT_SPEC, num_classes, weights_dir, pretrained=pretrained)


def forward_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        feat = model.forward_features(x)
    else:
        raise RuntimeError(f"{type(model).__name__} does not expose forward_features")
    if isinstance(feat, (list, tuple)):
        feat = feat[-1]
    if feat.ndim == 2:
        feat = feat[:, :, None, None]
    return feat


def pooled_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    feat = forward_features(model, x)
    if feat.ndim == 4:
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
    return feat
