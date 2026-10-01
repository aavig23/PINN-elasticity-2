"""Collocation points (random) and evaluation grids (deterministic), nondimensional.

Random sets are drawn uniformly over the true domain, so the 2:1 aspect ratio
is respected automatically (spec §5.6). Each draw uses its own generator
seeded from (run seed, set index), so a resumed run regenerates exactly the
same points. Points are drawn in float64 on the CPU, then cast, so float32 and
float64 runs (and CPU and GPU runs) see the same points.
"""

from dataclasses import dataclass

import torch

from .physics.domain import Domain

DIAGNOSTIC_STREAM = 999_983  # set index reserved for diagnostic samples


@dataclass
class PointSets:
    interior: torch.Tensor  # (N, d)
    edges: dict[str, torch.Tensor]  # edge name -> (N_e, d)
    initial: torch.Tensor | None  # (N_0, 3) at t = 0, soft Stage 2 only


def _generator(seed: int, index: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed * 1_000_003 + index)


def _uniform(n: int, domain: Domain, g: torch.Generator) -> torch.Tensor:
    lower = torch.tensor(domain.lower, dtype=torch.float64)
    upper = torch.tensor(domain.upper, dtype=torch.float64)
    return lower + (upper - lower) * torch.rand(n, domain.dim, dtype=torch.float64, generator=g)


def sample_points(domain: Domain, sampling, *, constraints_mode: str, seed: int, index: int, dtype, device) -> PointSets:
    """Draw the training point sets for set number ``index``.

    The clamped edge is sampled only in soft mode, and initial points only for
    a soft Stage 2 run; in hard mode those conditions hold exactly by construction.
    """
    g = _generator(seed, index)
    interior = _uniform(sampling.n_interior, domain, g)

    counts = {"top": sampling.n_top, "bottom": sampling.n_bottom, "right": sampling.n_right}
    if constraints_mode == "soft":
        counts["left"] = sampling.n_left
    edges = {}
    for edge, n in counts.items():
        pts = _uniform(n, domain, g)
        axis, value = domain.edge_coordinate(edge)
        pts[:, axis] = value
        edges[edge] = pts

    initial = None
    if constraints_mode == "soft" and domain.stage == 2:
        initial = _uniform(sampling.n_initial, domain, g)
        initial[:, 2] = 0.0

    cast = lambda t: t.to(device=device, dtype=dtype)  # noqa: E731
    return PointSets(
        interior=cast(interior),
        edges={k: cast(v) for k, v in edges.items()},
        initial=None if initial is None else cast(initial),
    )


def sample_interior(domain: Domain, n: int, *, seed: int, index: int, dtype, device) -> torch.Tensor:
    return _uniform(n, domain, _generator(seed, index)).to(device=device, dtype=dtype)


def grid_xz(nx: int, nz: int, z_max: float, times=None, *, dtype, device) -> torch.Tensor:
    """Regular grid over the plate: x fastest, then z, then t (if ``times`` given).

    Returns (nz*nx, 2) or (nt*nz*nx, 3). Reshape values to (nz, nx) or (nt, nz, nx).
    """
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    z = torch.linspace(0.0, z_max, nz, dtype=torch.float64)
    if times is None:
        zz, xx = torch.meshgrid(z, x, indexing="ij")
        pts = torch.stack([xx.reshape(-1), zz.reshape(-1)], dim=1)
    else:
        t = torch.as_tensor(times, dtype=torch.float64)
        tt, zz, xx = torch.meshgrid(t, z, x, indexing="ij")
        pts = torch.stack([xx.reshape(-1), zz.reshape(-1), tt.reshape(-1)], dim=1)
    return pts.to(device=device, dtype=dtype)


def edge_grid(domain: Domain, edge: str, n: int, n_times: int, *, exclude_clamped_corner: bool, dtype, device):
    """Regular points along one edge (times a regular time grid for Stage 2).

    With ``exclude_clamped_corner``, the top/bottom edges skip x = 0, where the
    clamp meets a traction-free edge and the solution is singular (spec §5.7).
    """
    axis, value = domain.edge_coordinate(edge)
    along_axis = 1 - axis
    along = torch.linspace(domain.lower[along_axis], domain.upper[along_axis], n, dtype=torch.float64)
    if exclude_clamped_corner and edge in ("top", "bottom"):
        along = along[1:]
    pts = torch.zeros(len(along), domain.dim, dtype=torch.float64)
    pts[:, along_axis] = along
    pts[:, axis] = value
    if domain.stage == 2:
        t = torch.linspace(0.0, domain.T_hat, n_times, dtype=torch.float64)
        pts = pts.repeat(n_times, 1)
        pts[:, 2] = t.repeat_interleave(len(along))
    return pts.to(device=device, dtype=dtype)
