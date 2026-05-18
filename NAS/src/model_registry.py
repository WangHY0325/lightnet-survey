from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import timm


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    source: str
    model_name: str
    paper_family: str
    weight_file: str = ""


MODEL_SPECS = [
    ModelSpec("mobilenetv2", "MobileNetV2", "timm", "mobilenetv2_100", "manual/mobile", "mobilenetv2_100_ra-b33bc2c4.pth"),
    ModelSpec("mnasnet", "MnasNet", "timm", "mnasnet_100", "hardware-aware NAS", "mnasnet_b1-74cb7081.pth"),
    ModelSpec("fbnet", "FBNet-C", "timm", "fbnetc_100", "differentiable hardware-aware NAS", "fbnetc_100-c345b898.pth"),
    ModelSpec("mobilenetv3_large", "MobileNetV3-Large", "timm", "mobilenetv3_large_100", "NAS plus manual operator design", "mobilenetv3_large_100_ra-f55367f5.pth"),
    ModelSpec("efficientnet_b0", "EfficientNet-B0", "timm", "efficientnet_b0", "compound-scaling NAS family", "efficientnet_b0_ra-3dd342df.pth"),
]


def parse_model_keys(raw: str | Iterable[str]) -> list[str]:
    if isinstance(raw, str):
        if raw.lower() == "all":
            return [spec.key for spec in MODEL_SPECS]
        keys = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        keys = list(raw)
    known = {spec.key for spec in MODEL_SPECS}
    missing = [key for key in keys if key not in known]
    if missing:
        raise ValueError(f"Unknown model keys {missing}; known keys: {sorted(known)}")
    return keys


def get_spec(key: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(key)


def create_model(key: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    spec = get_spec(key)
    if spec.source != "timm":
        raise ValueError(f"Unsupported source: {spec.source}")
    return _load_timm(spec, num_classes, pretrained)


def _load_timm(spec: ModelSpec, num_classes: int, pretrained: bool) -> nn.Module:
    if not pretrained:
        return timm.create_model(spec.model_name, pretrained=False, num_classes=num_classes)
    repo_root = Path(__file__).resolve().parents[1]
    weight_path = repo_root / "weights" / spec.weight_file
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
    _replace_classifier(model, num_classes)
    return model


def _replace_classifier(model: nn.Module, num_classes: int) -> None:
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes)
        return
    parent, name, layer = _find_last_linear(model)
    if parent is None or name is None or layer is None:
        raise RuntimeError("Could not find a Linear classifier to replace")
    setattr(parent, name, nn.Linear(layer.in_features, num_classes))


def _find_last_linear(model: nn.Module):
    last = (None, None, None)
    for _, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                last = (module, child_name, child)
    return last


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
