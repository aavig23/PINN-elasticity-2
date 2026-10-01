"""Training-health diagnostics (plan §14). All nondimensional.

These describe how well the trained network satisfies its own equations and
conditions. They are not validation: nothing here is compared with an
analytical, FEM or other reference solution (spec §6).
"""

import copy
import math

import torch

from ..physics.boundary import INITIAL_CONDITIONS, condition_residual, conditions_for_edge
from ..physics.equations import evaluate
from ..sampling import edge_grid, grid_xz


def corner_mask(X: torch.Tensor, radius: float, z_max: float) -> torch.Tensor:
    """True for points within ``radius`` of a clamped corner (0, 0) or (0, z_max)."""
    x, z = X[:, 0], X[:, 1]
    near_bottom = torch.sqrt(x**2 + z**2) < radius
    near_top = torch.sqrt(x**2 + (z - z_max) ** 2) < radius
    return near_bottom | near_top


def pde_excluding_corners(fields: dict, radius: float, z_max: float) -> float:
    """mean(R_u^2) + mean(R_w^2) over points outside the clamped-corner neighbourhoods.

    The corners are singular (spec §5.7), so this shows the residual away from them.
    """
    keep = ~corner_mask(fields["X"], radius, z_max)
    if not bool(keep.any()):
        return math.nan
    return float(fields["R_u"][keep].square().mean() + fields["R_w"][keep].square().mean())


def _evaluate_detached(model, X, scales, stage, residual, chunk=8192) -> dict:
    parts: dict[str, list] = {}
    for i in range(0, X.shape[0], chunk):
        f = evaluate(model, X[i : i + chunk], scales, stage=stage, residual=residual)
        for key, value in f.items():
            parts.setdefault(key, []).append(value.detach())
    return {key: torch.cat(values) for key, values in parts.items()}


def _max_abs(t: torch.Tensor) -> float:
    return float(t.abs().max())


def _rms(t: torch.Tensor) -> float:
    return float(t.square().mean().sqrt())


def precision_check(model, X: torch.Tensor, scales, stage: int) -> dict:
    """Residuals of the same network in float32 and float64 at the same points (decision B2)."""
    results = {}
    for dtype in (torch.float32, torch.float64):
        m = copy.deepcopy(model).to(dtype)
        f = _evaluate_detached(m, X.to(dtype), scales, stage, residual=True)
        results[dtype] = torch.cat([f["R_u"], f["R_w"]]).to(torch.float64)
    r32, r64 = results[torch.float32], results[torch.float64]
    rms64, diff = _rms(r64), _rms(r32 - r64)
    return {
        "n_points": int(X.shape[0]),
        "rms_residual_float64": rms64,
        "rms_diff_float32_vs_float64": diff,
        "relative_diff": diff / rms64 if rms64 > 0 else math.nan,
    }


def run_diagnostics(model, cfg, scales, domain, loading) -> dict:
    """Residual, traction-BC and constraint diagnostics on regular grids."""
    p = next(model.parameters())
    dtype, device, stage = p.dtype, p.device, domain.stage
    d, radius = cfg.diagnostics, cfg.loss.corner_exclusion_radius
    times = cfg.snapshot_times() if stage == 2 else None

    # PDE residual on the output grid (at the snapshot times for Stage 2).
    X = grid_xz(cfg.output.nx, cfg.output.nz, domain.z_max, times, dtype=dtype, device=device)
    f = _evaluate_detached(model, X, scales, stage, residual=True)
    residual = {
        "n_points": int(X.shape[0]),
        "mean_sq_u": float(f["R_u"].square().mean()),
        "mean_sq_w": float(f["R_w"].square().mean()),
        "max_abs_u": _max_abs(f["R_u"]),
        "max_abs_w": _max_abs(f["R_w"]),
        "mean_sq_excl_corners": pde_excluding_corners(f, radius, domain.z_max),
    }

    # Traction conditions on dense edge grids (clamped corners excluded).
    traction = {}
    for edge in ("top", "bottom", "right"):
        Xe = edge_grid(domain, edge, d.edge_points, d.edge_times, exclude_clamped_corner=True, dtype=dtype, device=device)
        fe = _evaluate_detached(model, Xe, scales, stage, residual=False)
        for cond in conditions_for_edge(edge):
            r = condition_residual(cond, fe, loading)
            traction[f"{edge}_{cond.quantity}_max_abs"] = _max_abs(r)
            traction[f"{edge}_{cond.quantity}_rms"] = _rms(r)

    # Clamp and initial conditions: exactly 0 in hard mode, penalised in soft mode.
    constraints = {}
    Xl = edge_grid(domain, "left", d.edge_points, d.edge_times, exclude_clamped_corner=False, dtype=dtype, device=device)
    fl = _evaluate_detached(model, Xl, scales, stage, residual=False)
    constraints["clamp_u_max_abs"] = _max_abs(fl["u"])
    constraints["clamp_w_max_abs"] = _max_abs(fl["w"])
    if stage == 2:
        X0 = grid_xz(cfg.output.nx, cfg.output.nz, domain.z_max, [0.0], dtype=dtype, device=device)
        f0 = _evaluate_detached(model, X0, scales, stage, residual=False)
        for cond in INITIAL_CONDITIONS:
            constraints[f"ic_{cond.quantity}_max_abs"] = _max_abs(f0[cond.quantity])

    return {"residual": residual, "traction_bc": traction, "constraints": constraints}
