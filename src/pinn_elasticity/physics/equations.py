"""Equilibrium / equations of motion, nondimensional (spec §2.3, §3.2).

    Dynamic: R_u = inertia_hat * u_tt - (d sxx/dx + d sxz/dz)
             R_w = inertia_hat * w_tt - (d sxz/dx + d szz/dz)
    Static:  the same with the inertia terms removed.
No body force. inertia_hat = 1 with the spec scaling.
"""

import torch

from ..utils.autodiff import gradient
from .constitutive import plane_stress_stresses
from .scaling import Scales


def evaluate(model, X: torch.Tensor, scales: Scales, *, stage: int, residual: bool) -> dict:
    """Run the model at points X and derive displacements, gradients, stresses
    and (if ``residual``) the PDE residuals, all nondimensional.

    X: (N, 2) = (x_hat, z_hat) for stage 1, (N, 3) = (x_hat, z_hat, t_hat) for stage 2.
    Returns a dict of (N,) tensors: u, w, u_x, u_z, w_x, w_z, [u_t, w_t],
    sxx, szz, sxz, [R_u, R_w], plus the points as "X" (with requires_grad).
    """
    expected_dim = 2 if stage == 1 else 3
    if X.ndim != 2 or X.shape[1] != expected_dim:
        raise ValueError(f"stage {stage} expects points of shape (N, {expected_dim}), got {tuple(X.shape)}")

    X = X.detach().requires_grad_(True)
    out = model(X)
    u, w = out[:, 0], out[:, 1]
    du, dw = gradient(u, X), gradient(w, X)
    f = {"X": X, "u": u, "w": w, "u_x": du[:, 0], "u_z": du[:, 1], "w_x": dw[:, 0], "w_z": dw[:, 1]}
    if stage == 2:
        f["u_t"], f["w_t"] = du[:, 2], dw[:, 2]

    sxx, szz, sxz = plane_stress_stresses(
        f["u_x"], f["u_z"], f["w_x"], f["w_z"], lam_star=scales.lam_hat, mu=scales.mu_hat
    )
    f.update(sxx=sxx, szz=szz, sxz=sxz)

    if residual:
        div_x = gradient(sxx, X)[:, 0] + gradient(sxz, X)[:, 1]
        div_z = gradient(sxz, X)[:, 0] + gradient(szz, X)[:, 1]
        if stage == 2:
            u_tt = gradient(f["u_t"], X)[:, 2]
            w_tt = gradient(f["w_t"], X)[:, 2]
            f["R_u"] = scales.inertia_hat * u_tt - div_x
            f["R_w"] = scales.inertia_hat * w_tt - div_z
        else:
            f["R_u"] = -div_x
            f["R_w"] = -div_z
    return f
