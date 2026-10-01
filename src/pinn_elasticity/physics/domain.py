"""Nondimensional problem domain (spec §1.1, §3.2).

    x_hat in [0, 1], z_hat in [0, H/L] = [0, 0.5], and for Stage 2 t_hat in [0, T_hat].
Column order of every point tensor: (x_hat, z_hat) or (x_hat, z_hat, t_hat).
"""

from dataclasses import dataclass

from .scaling import Scales

# Edge name -> (column index, "min" | "max") of the coordinate held fixed.
EDGES = {
    "left": (0, "min"),  # x = 0
    "right": (0, "max"),  # x = L
    "bottom": (1, "min"),  # z = 0
    "top": (1, "max"),  # z = H
}


@dataclass(frozen=True)
class Domain:
    stage: int
    z_max: float
    T_hat: float | None  # None for Stage 1 (static)
    x_max: float = 1.0

    @classmethod
    def from_config(cls, cfg, scales: Scales) -> "Domain":
        stage = cfg.experiment.stage
        return cls(stage=stage, z_max=scales.z_hat_max, T_hat=cfg.time.T_hat if stage == 2 else None)

    @property
    def dim(self) -> int:
        return 2 if self.stage == 1 else 3

    @property
    def lower(self) -> tuple[float, ...]:
        return (0.0, 0.0) if self.stage == 1 else (0.0, 0.0, 0.0)

    @property
    def upper(self) -> tuple[float, ...]:
        if self.stage == 1:
            return (self.x_max, self.z_max)
        return (self.x_max, self.z_max, self.T_hat)

    def edge_coordinate(self, edge: str) -> tuple[int, float]:
        """(column, value) of the coordinate that is fixed on ``edge``."""
        axis, side = EDGES[edge]
        return axis, (self.lower if side == "min" else self.upper)[axis]

    @property
    def tip(self) -> tuple[float, float]:
        """Tip point (L, H/2) in nondimensional coordinates."""
        return (self.x_max, self.z_max / 2.0)

    @property
    def clamped_corners(self) -> tuple[tuple[float, float], ...]:
        """Corners where the clamp meets a traction-free edge (singular, spec §5.7)."""
        return ((0.0, 0.0), (0.0, self.z_max))
