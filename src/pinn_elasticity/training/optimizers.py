"""Optimizer schedule: Adam with exponential decay, then L-BFGS (decision B4)."""

import torch


def make_adam(params, lr: float) -> torch.optim.Adam:
    return torch.optim.Adam(params, lr=lr)


def make_adam_scheduler(optimizer, lr_start: float, lr_final: float, iterations: int):
    """Exponential decay from lr_start to lr_final over ``iterations`` steps."""
    gamma = (lr_final / lr_start) ** (1.0 / max(iterations, 1))
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


def make_lbfgs(params, history: int, line_search: str) -> torch.optim.LBFGS:
    # max_iter=1: one L-BFGS iteration per step() call, so the trainer can log
    # and checkpoint between iterations. The curvature history lives in the
    # optimizer state and carries over between calls.
    return torch.optim.LBFGS(params, lr=1.0, max_iter=1, history_size=history, line_search_fn=line_search)
