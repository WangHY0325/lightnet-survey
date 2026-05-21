import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Fix all RNGs for reproducibility on CPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep determinism conservative; CPU run anyway.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
