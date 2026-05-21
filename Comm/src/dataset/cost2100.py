"""COST2100 CSI feedback dataset loader.

Each .mat file from the Wen & Jin pre-processed COST2100 release contains a
single key ``HT`` of shape ``(N, 2*32*32)``. We reshape to ``(N, 2, 32, 32)``
and wrap with TensorDataset. The convention in this dataset is that values
are pre-normalized to [0, 1]; that is why CRissNet/CLNet apply Sigmoid at
the decoder output and centralize by 0.5 inside the NMSE evaluator.
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, TensorDataset


__all__ = ["load_split", "Cost2100DataLoader"]


_CHANNEL = 2
_NT = 32
_NC = 32


def _filename(scenario: str, split: str) -> str:
    """Build the canonical .mat filename used by the upstream release.

    The upstream layout is ``DATA_H{split}{scenario}.mat`` where split is one
    of {train, val, test} and scenario is one of {in, out}.
    """
    if scenario not in ("in", "out"):
        raise ValueError(f"scenario must be 'in' or 'out', got {scenario!r}")
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', 'test', got {split!r}")
    return f"DATA_H{split}{scenario}.mat"


def load_split(root: str, scenario: str, split: str) -> torch.Tensor:
    """Load one split as a ``(N, 2, 32, 32)`` float32 tensor."""
    path = Path(root) / _filename(scenario, split)
    if not path.exists():
        raise FileNotFoundError(
            f"COST2100 file not found: {path}. "
            "Place the six DATA_H*.mat files under data/cost2100/."
        )
    mat = sio.loadmat(str(path))
    if "HT" not in mat:
        raise KeyError(f"{path}: expected key 'HT', found {list(mat.keys())}")
    arr = np.asarray(mat["HT"], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != _CHANNEL * _NT * _NC:
        raise ValueError(
            f"{path}: expected HT of shape (N, {_CHANNEL * _NT * _NC}), got {arr.shape}"
        )
    arr = arr.reshape(arr.shape[0], _CHANNEL, _NT, _NC)
    return torch.from_numpy(arr)


class Cost2100DataLoader:
    """Train/Val/Test loaders for one (scenario) configuration."""

    def __init__(
        self,
        root: str,
        scenario: str,
        batch_size: int = 200,
        num_workers: int = 4,
        pin_memory: bool = False,
    ) -> None:
        self.root = root
        self.scenario = scenario
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train = load_split(root, scenario, "train")
        self.val = load_split(root, scenario, "val")
        self.test = load_split(root, scenario, "test")

    def __call__(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = TensorDataset(self.train)
        val_ds = TensorDataset(self.val)
        test_ds = TensorDataset(self.test)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False,
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False,
        )
        return train_loader, val_loader, test_loader
