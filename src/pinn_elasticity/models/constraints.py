"""Hard or soft enforcement of the clamp and initial conditions (spec §5.4, decision B3).

Hard mode multiplies the network output N(x, z[, t]) by a factor that vanishes
where the condition applies:
    Stage 1: (u, w) = x_hat * N                 -> u = w = 0 at x = 0 exactly
    Stage 2: (u, w) = x_hat * (t_hat/T_hat)^2 * N -> also u = w = u_t = w_t = 0 at t = 0
(t_hat/T_hat)^2 is the spec's t_hat^2 divided by a constant, so it keeps the
factor in [0, 1] without changing the function class.
Traction conditions are always soft. Soft mode returns N unchanged and the
clamp/initial conditions are enforced by penalty losses instead.
"""

import torch
from torch import nn

from ..physics.domain import Domain
from .networks import MLP


class ConstrainedPINN(nn.Module):
    def __init__(self, net: nn.Module, *, mode: str, stage: int, T_hat: float | None):
        super().__init__()
        if mode not in ("hard", "soft"):
            raise ValueError(f"unknown constraints mode {mode!r}")
        self.net, self.mode, self.stage, self.T_hat = net, mode, stage, T_hat

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.net(X)
        if self.mode == "soft":
            return out
        factor = X[:, 0:1]
        if self.stage == 2:
            factor = factor * (X[:, 2:3] / self.T_hat) ** 2
        return factor * out


def build_pinn(cfg, domain: Domain) -> ConstrainedPINN:
    net = MLP(domain.lower, domain.upper, cfg.model.hidden_layers, cfg.model.width)
    return ConstrainedPINN(net, mode=cfg.constraints.mode, stage=domain.stage, T_hat=domain.T_hat)
