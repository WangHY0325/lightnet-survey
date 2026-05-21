from .csinet import CsiNet, build_csinet
from .crissnet import CRissNet, build_crissnet


def build_model(name: str, reduction: int) -> "torch.nn.Module":
    """Factory for CSI feedback models.

    Args:
        name: 'csinet' or 'crissnet'.
        reduction: compression ratio denominator (4, 8, 16, 32, 64); the
            codeword length is ``2*32*32 // reduction``.
    """
    name = name.lower()
    if name == "csinet":
        return build_csinet(reduction=reduction)
    if name == "crissnet":
        return build_crissnet(reduction=reduction)
    raise ValueError(f"unknown model name: {name!r}")


__all__ = ["CsiNet", "CRissNet", "build_csinet", "build_crissnet", "build_model"]
