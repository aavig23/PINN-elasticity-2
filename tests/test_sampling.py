import pytest
import torch

from pinn_elasticity.physics.domain import Domain
from pinn_elasticity.sampling import edge_grid, grid_xz, sample_points

D1 = Domain(stage=1, z_max=0.5, T_hat=None)
D2 = Domain(stage=2, z_max=0.5, T_hat=4.0)


def _sample(make_config, stage, mode="hard", index=0, dtype=torch.float64):
    cfg = make_config(stage, {"constraints": {"mode": mode}})
    domain = D1 if stage == 1 else D2
    return cfg, sample_points(domain, cfg.sampling, constraints_mode=mode, seed=0, index=index, dtype=dtype, device="cpu")


@pytest.mark.parametrize("stage", [1, 2])
def test_counts_shapes_and_bounds(make_config, stage):
    cfg, p = _sample(make_config, stage)
    s, dim = cfg.sampling, stage + 1
    assert p.interior.shape == (s.n_interior, dim)
    assert {k: v.shape for k, v in p.edges.items()} == {
        "top": (s.n_top, dim),
        "bottom": (s.n_bottom, dim),
        "right": (s.n_right, dim),
    }
    upper = torch.tensor(D1.upper if stage == 1 else D2.upper, dtype=torch.float64)
    for pts in [p.interior, *p.edges.values()]:
        assert torch.all(pts >= 0) and torch.all(pts <= upper)


def test_edge_points_lie_exactly_on_their_edges(make_config):
    _, p = _sample(make_config, 2, mode="soft")
    assert torch.all(p.edges["top"][:, 1] == 0.5)
    assert torch.all(p.edges["bottom"][:, 1] == 0.0)
    assert torch.all(p.edges["right"][:, 0] == 1.0)
    assert torch.all(p.edges["left"][:, 0] == 0.0)
    assert torch.all(p.initial[:, 2] == 0.0)


def test_hard_mode_skips_clamp_and_initial_sets(make_config):
    _, p = _sample(make_config, 2, mode="hard")
    assert "left" not in p.edges and p.initial is None
    _, p1 = _sample(make_config, 1, mode="soft")
    assert "left" in p1.edges and p1.initial is None  # Stage 1 has no initial conditions


def test_sampling_is_reproducible_and_index_dependent(make_config):
    _, a = _sample(make_config, 1, index=3)
    _, b = _sample(make_config, 1, index=3)
    _, c = _sample(make_config, 1, index=4)
    assert torch.equal(a.interior, b.interior)
    assert not torch.equal(a.interior, c.interior)


def test_float32_and_float64_see_the_same_points(make_config):
    _, a = _sample(make_config, 1, dtype=torch.float32)
    _, b = _sample(make_config, 1, dtype=torch.float64)
    assert a.interior.dtype == torch.float32
    assert torch.equal(a.interior, b.interior.float())


def test_grids():
    g1 = grid_xz(11, 6, 0.5, dtype=torch.float64, device="cpu")
    assert g1.shape == (66, 2) and g1[-1].tolist() == [1.0, 0.5]
    g2 = grid_xz(11, 6, 0.5, times=[1.0, 2.0], dtype=torch.float64, device="cpu")
    assert g2.shape == (132, 3) and set(g2[:, 2].tolist()) == {1.0, 2.0}
    top = edge_grid(D2, "top", 21, 5, exclude_clamped_corner=True, dtype=torch.float64, device="cpu")
    assert top.shape == (20 * 5, 3) and torch.all(top[:, 1] == 0.5) and torch.all(top[:, 0] > 0)
