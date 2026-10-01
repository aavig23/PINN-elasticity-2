"""Right-edge axial load (spec §2.5, §4).

q0 = P0 / H with P0 a force per unit out-of-plane thickness, so q0 is a traction [Pa].
Stage 1: q = q0. Stage 2: q = q0 * f(t_hat) with the smooth ramp
    f = 0.5 (1 - cos(pi t_hat / t_r_hat)) for t_hat < t_r_hat, else 1.
"""

import math

import torch

from .scaling import Scales


def edge_traction_q0(P0: float, H: float) -> float:
    return P0 / H


def ramp(t_hat: torch.Tensor, t_r_hat: float) -> torch.Tensor:
    rising = 0.5 * (1.0 - torch.cos(math.pi * t_hat / t_r_hat))
    return torch.where(t_hat < t_r_hat, rising, torch.ones_like(t_hat))


class Loading:
    def __init__(self, scales: Scales, stage: int, t_r_hat: float):
        self.stage = stage
        self.t_r_hat = t_r_hat
        self.q0 = edge_traction_q0(scales.P0, scales.H)
        # Nondimensional load amplitude q0 / sigma_c (exactly 1 with the spec scaling).
        self.amplitude_hat = self.q0 / scales.sigma_c

    def q_hat(self, X: torch.Tensor) -> torch.Tensor:
        """Nondimensional traction sigma_xx target at right-edge points X, shape (N,)."""
        if self.stage == 1:
            return torch.full((X.shape[0],), self.amplitude_hat, dtype=X.dtype, device=X.device)
        return self.amplitude_hat * ramp(X[:, 2], self.t_r_hat)
