"""Device and dtype selection (spec §5.1: GPU on Colab, CPU fallback)."""

import platform

import torch

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def resolve_dtype(name: str) -> torch.dtype:
    return DTYPES[name]


def select_device(preferred: str | None = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine() or "cpu"
