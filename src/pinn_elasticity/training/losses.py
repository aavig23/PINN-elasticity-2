"""Loss terms and total loss (spec §5.5). All terms are nondimensional mean squares.

    L_total = w_pde (pde_u + pde_w) + w_free (free_zz + free_xz) + w_load (load_xx + load_xz)
            + w_clamp (clamp_u + clamp_w)            [soft mode]
            + w_ic (ic_u + ic_w + ic_ut + ic_wt)     [soft mode, Stage 2]
free_* are means over the union of top and bottom points (spec: "on top and bottom").
"""

import torch

from ..physics.boundary import INITIAL_CONDITIONS, condition_residual, conditions_for_edge
from ..physics.equations import evaluate

TERM_GROUPS = {
    "pde": ("pde_u", "pde_w"),
    "free": ("free_zz", "free_xz"),
    "load": ("load_xx", "load_xz"),
    "clamp": ("clamp_u", "clamp_w"),
    "ic": ("ic_u", "ic_w", "ic_ut", "ic_wt"),
}


def active_groups(stage: int, mode: str) -> tuple[str, ...]:
    groups = ["pde", "free", "load"]
    if mode == "soft":
        groups.append("clamp")
        if stage == 2:
            groups.append("ic")
    return tuple(groups)


def active_terms(stage: int, mode: str) -> tuple[str, ...]:
    return tuple(t for g in active_groups(stage, mode) for t in TERM_GROUPS[g])


def compute_loss_terms(model, points, scales, loading, stage: int, mode: str):
    """Return (terms, interior).

    terms: unweighted scalar loss terms (with graph, ready for backward).
    interior: detached {"X", "R_u", "R_w"} at the interior points, for diagnostics.
    """
    interior = evaluate(model, points.interior, scales, stage=stage, residual=True)
    terms = {"pde_u": interior["R_u"].square().mean(), "pde_w": interior["R_w"].square().mean()}

    parts: dict[str, list[torch.Tensor]] = {}
    for edge, X in points.edges.items():
        fields = evaluate(model, X, scales, stage=stage, residual=False)
        for cond in conditions_for_edge(edge):
            parts.setdefault(cond.term, []).append(condition_residual(cond, fields, loading))
    if points.initial is not None:
        fields = evaluate(model, points.initial, scales, stage=stage, residual=False)
        for cond in INITIAL_CONDITIONS:
            parts.setdefault(cond.term, []).append(condition_residual(cond, fields, loading))
    for term, values in parts.items():
        terms[term] = torch.cat(values).square().mean()

    expected = active_terms(stage, mode)
    if set(terms) != set(expected):
        raise RuntimeError(f"point sets give loss terms {sorted(terms)}, expected {sorted(expected)}")
    diagnostics = {k: interior[k].detach() for k in ("X", "R_u", "R_w")}
    return {t: terms[t] for t in expected}, diagnostics


def weighted_total(terms: dict, weights: dict) -> torch.Tensor:
    """sum over groups of weight * (sum of the group's terms)."""
    total = None
    for group, weight in weights.items():
        for term in TERM_GROUPS[group]:
            contribution = weight * terms[term]
            total = contribution if total is None else total + contribution
    return total


def group_values(terms: dict, groups) -> dict[str, float]:
    """Sum of each group's (unweighted) terms as plain floats."""
    return {g: sum(float(terms[t].detach()) for t in TERM_GROUPS[g]) for g in groups}
