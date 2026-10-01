import pytest
import torch

from pinn_elasticity.models.constraints import build_pinn
from pinn_elasticity.physics.domain import Domain
from pinn_elasticity.physics.loading import Loading
from pinn_elasticity.physics.scaling import Scales
from pinn_elasticity.sampling import sample_points
from pinn_elasticity.training.losses import active_terms, compute_loss_terms, weighted_total
from pinn_elasticity.training.weighting import build_weighting

NU = 0.30


class Uniaxial(torch.nn.Module):
    def forward(self, X):
        return torch.stack([X[:, 0], -NU * X[:, 1]], dim=1)


def _setup(make_config, stage, mode):
    cfg = make_config(stage, {"constraints": {"mode": mode}})
    scales = Scales.from_config(cfg)
    domain = Domain.from_config(cfg, scales)
    loading = Loading(scales, stage, cfg.time.t_r_hat)
    points = sample_points(
        domain, cfg.sampling, constraints_mode=mode, seed=0, index=0, dtype=torch.float64, device="cpu"
    )
    return cfg, scales, domain, loading, points


def test_exact_uniaxial_field_zeroes_pde_traction_and_load_terms(make_config):
    cfg, scales, domain, loading, points = _setup(make_config, 1, "soft")
    terms, _ = compute_loss_terms(Uniaxial(), points, scales, loading, 1, "soft")
    for name in ("pde_u", "pde_w", "free_zz", "free_xz", "load_xx", "load_xz", "clamp_u"):
        assert float(terms[name].detach()) < 1e-24, name
    # The uniaxial field is not clamped in w: clamp_w = mean((nu z)^2) over left-edge points.
    z_left = points.edges["left"][:, 1]
    assert float(terms["clamp_w"].detach()) == pytest.approx(float((NU * z_left).square().mean()), rel=1e-12)


@pytest.mark.parametrize(
    "stage, mode, expected",
    [
        (1, "hard", {"pde_u", "pde_w", "free_zz", "free_xz", "load_xx", "load_xz"}),
        (1, "soft", {"pde_u", "pde_w", "free_zz", "free_xz", "load_xx", "load_xz", "clamp_u", "clamp_w"}),
        (2, "hard", {"pde_u", "pde_w", "free_zz", "free_xz", "load_xx", "load_xz"}),
        (
            2,
            "soft",
            {"pde_u", "pde_w", "free_zz", "free_xz", "load_xx", "load_xz", "clamp_u", "clamp_w"}
            | {"ic_u", "ic_w", "ic_ut", "ic_wt"},
        ),
    ],
)
def test_terms_present_per_stage_and_mode(make_config, stage, mode, expected):
    cfg, scales, domain, loading, points = _setup(make_config, stage, mode)
    assert set(active_terms(stage, mode)) == expected
    torch.manual_seed(0)
    model = build_pinn(cfg, domain).double()
    terms, interior = compute_loss_terms(model, points, scales, loading, stage, mode)
    assert set(terms) == expected
    assert all(t.ndim == 0 and torch.isfinite(t) for t in terms.values())
    assert interior["R_u"].shape == (cfg.sampling.n_interior,)


def test_total_loss_is_weighted_group_sum_and_backpropagates(make_config):
    cfg, scales, domain, loading, points = _setup(make_config, 2, "soft")
    torch.manual_seed(0)
    model = build_pinn(cfg, domain).double()
    terms, _ = compute_loss_terms(model, points, scales, loading, 2, "soft")
    weights = {"pde": 2.0, "free": 3.0, "load": 0.5, "clamp": 1.0, "ic": 4.0}
    manual = (
        2.0 * (terms["pde_u"] + terms["pde_w"])
        + 3.0 * (terms["free_zz"] + terms["free_xz"])
        + 0.5 * (terms["load_xx"] + terms["load_xz"])
        + 1.0 * (terms["clamp_u"] + terms["clamp_w"])
        + 4.0 * (terms["ic_u"] + terms["ic_w"] + terms["ic_ut"] + terms["ic_wt"])
    )
    total = weighted_total(terms, weights)
    assert torch.isclose(total, manual, rtol=1e-14)
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_fixed_weighting_returns_config_weights_for_active_groups(make_config):
    cfg = make_config(1, {"loss": {"weights": {"pde": 5.0}}})
    w = build_weighting(cfg.loss, ("pde", "free", "load"))
    assert w({}, 0) == {"pde": 5.0, "free": 1.0, "load": 1.0}
