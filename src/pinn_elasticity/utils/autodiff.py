"""Derivatives of per-point network outputs with respect to the input points."""

import torch


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d y_i / d x_i for every point i.

    y: shape (N,), one scalar per point; x: shape (N, d) with requires_grad.
    Returns shape (N, d). Points are independent, so differentiating sum(y)
    gives each point's own gradient. create_graph=True keeps the result
    differentiable, which second derivatives and backpropagation to the
    network weights both need (spec §5.1, §5.7).
    """
    if not y.requires_grad:
        return torch.zeros_like(x)
    (g,) = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, allow_unused=True)
    return torch.zeros_like(x) if g is None else g
