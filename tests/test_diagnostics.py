import math

import torch

from pinn_elasticity.models.constraints import build_pinn
from pinn_elasticity.physics.domain import Domain
from pinn_elasticity.physics.loading import Loading
from pinn_elasticity.physics.scaling import Scales
from pinn_elasticity.sampling import sample_interior
from pinn_elasticity.training.diagnostics import corner_mask, precision_check, run_diagnostics


def _setup(make_config, stage, mode="hard"):
    cfg = make_config(stage, {"constraints": {"mode": mode}})
    scales = Scales.from_config(cfg)
    domain = Domain.from_config(cfg, scales)
    torch.manual_seed(0)
    model = build_pinn(cfg, domain).double()
    return cfg, scales, domain, Loading(scales, stage, cfg.time.t_r_hat), model


def test_corner_mask():
    X = torch.tensor([[0.01, 0.01], [0.01, 0.49], [0.5, 0.25], [1.0, 0.0]], dtype=torch.float64)
    assert corner_mask(X, 0.05, 0.5).tolist() == [True, True, False, False]


def test_precision_check_is_small_for_a_fresh_network(make_config):
    cfg, scales, domain, _, model = _setup(make_config, 2)
    X = sample_interior(domain, 256, seed=0, index=0, dtype=torch.float64, device="cpu")
    result = precision_check(model, X, scales, 2)
    assert result["rms_residual_float64"] > 0
    assert result["relative_diff"] < 1e-3


def test_hard_mode_constraints_are_exactly_zero(make_config):
    for stage in (1, 2):
        cfg, scales, domain, loading, model = _setup(make_config, stage)
        d = run_diagnostics(model, cfg, scales, domain, loading)
        assert all(v == 0.0 for v in d["constraints"].values()), d["constraints"]
        expected = {"clamp_u_max_abs", "clamp_w_max_abs"}
        if stage == 2:
            expected |= {"ic_u_max_abs", "ic_w_max_abs", "ic_u_t_max_abs", "ic_w_t_max_abs"}
        assert set(d["constraints"]) == expected


def test_diagnostics_are_finite_and_complete(make_config):
    cfg, scales, domain, loading, model = _setup(make_config, 2, mode="soft")
    d = run_diagnostics(model, cfg, scales, domain, loading)
    assert d["residual"]["n_points"] == cfg.output.nx * cfg.output.nz * len(cfg.snapshot_times())
    assert {"top_szz_max_abs", "bottom_sxz_rms", "right_sxx_max_abs", "right_sxz_max_abs"} <= set(d["traction_bc"])
    values = [*d["residual"].values(), *d["traction_bc"].values(), *d["constraints"].values()]
    assert all(math.isfinite(v) for v in values)
    assert d["constraints"]["clamp_u_max_abs"] > 0  # soft mode: not exact before training
