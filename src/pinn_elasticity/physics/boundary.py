"""Boundary and initial conditions, transcribed from spec §2.4 and §2.6.

Each condition says which field must equal which target on which edge, and
which loss term it feeds (spec §5.5). Losses and diagnostics are driven from
these tables, so the conditions are written down in exactly one place.
"""

from dataclasses import dataclass

ZERO = "zero"
LOAD = "load"  # the right-edge traction q_hat (physics/loading.py)


@dataclass(frozen=True)
class Condition:
    edge: str  # "left" | "right" | "bottom" | "top" | "initial"
    quantity: str  # field key from physics.equations.evaluate
    target: str  # ZERO or LOAD
    term: str  # loss term the condition contributes to


# Spec §2.4 (both stages).
BOUNDARY_CONDITIONS = (
    Condition("left", "u", ZERO, "clamp_u"),  # clamped: u = 0
    Condition("left", "w", ZERO, "clamp_w"),  #          w = 0
    Condition("bottom", "szz", ZERO, "free_zz"),  # traction-free
    Condition("bottom", "sxz", ZERO, "free_xz"),
    Condition("top", "szz", ZERO, "free_zz"),  # traction-free
    Condition("top", "sxz", ZERO, "free_xz"),
    Condition("right", "sxx", LOAD, "load_xx"),  # axial tension: sxx = q
    Condition("right", "sxz", ZERO, "load_xz"),  #                sxz = 0
)

# Spec §2.6 (Stage 2 only): plate at rest at t = 0.
INITIAL_CONDITIONS = (
    Condition("initial", "u", ZERO, "ic_u"),
    Condition("initial", "w", ZERO, "ic_w"),
    Condition("initial", "u_t", ZERO, "ic_ut"),
    Condition("initial", "w_t", ZERO, "ic_wt"),
)


def conditions_for_edge(edge: str) -> tuple[Condition, ...]:
    return tuple(c for c in BOUNDARY_CONDITIONS if c.edge == edge)


def condition_residual(condition: Condition, fields: dict, loading):
    """Field value minus its target at every point, shape (N,)."""
    value = fields[condition.quantity]
    if condition.target == LOAD:
        return value - loading.q_hat(fields["X"])
    return value
