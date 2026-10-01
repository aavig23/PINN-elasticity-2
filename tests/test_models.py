import torch

from pinn_elasticity.models.constraints import ConstrainedPINN, build_pinn
from pinn_elasticity.models.networks import MLP
from pinn_elasticity.physics.domain import Domain
from pinn_elasticity.utils.autodiff import gradient


def _model(make_config, stage, mode="hard"):
    cfg = make_config(stage, {"constraints": {"mode": mode}})
    domain = Domain(stage=stage, z_max=0.5, T_hat=4.0 if stage == 2 else None)
    torch.manual_seed(0)
    return build_pinn(cfg, domain).double()


def test_mlp_shapes_and_finite_outputs():
    net = MLP((0.0, 0.0, 0.0), (1.0, 0.5, 4.0), hidden_layers=3, width=8).double()
    out = net(torch.rand(17, 3, dtype=torch.float64))
    assert out.shape == (17, 2) and torch.isfinite(out).all()


def test_input_normalisation_maps_domain_to_unit_box():
    net = MLP((0.0, 0.0, 0.0), (1.0, 0.5, 4.0), hidden_layers=1, width=4)
    corners = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, 4.0]])
    assert torch.equal(net.normalize(corners), torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))


def test_hard_clamp_is_exact_stage1(make_config):
    model = _model(make_config, 1)
    X = torch.rand(50, 2, dtype=torch.float64)
    X[:, 0] = 0.0
    assert torch.all(model(X) == 0.0)


def test_hard_clamp_and_initial_conditions_are_exact_stage2(make_config):
    model = _model(make_config, 2)
    X = torch.rand(50, 3, dtype=torch.float64)
    X[:, 1] *= 0.5
    left = X.clone()
    left[:, 0] = 0.0
    assert torch.all(model(left) == 0.0)

    t0 = X.clone()
    t0[:, 2] = 0.0
    t0.requires_grad_(True)
    out = model(t0)
    assert torch.all(out == 0.0)
    assert torch.all(gradient(out[:, 0], t0)[:, 2] == 0.0)  # u_t = 0
    assert torch.all(gradient(out[:, 1], t0)[:, 2] == 0.0)  # w_t = 0


def test_soft_mode_is_identity(make_config):
    model = _model(make_config, 2, mode="soft")
    X = torch.rand(10, 3, dtype=torch.float64)
    assert torch.equal(model(X), model.net(X))


def test_gradients_reach_every_parameter(make_config):
    model = _model(make_config, 2)
    X = torch.rand(10, 3, dtype=torch.float64)
    model(X).square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_unknown_mode_rejected():
    import pytest

    with pytest.raises(ValueError):
        ConstrainedPINN(torch.nn.Identity(), mode="plane_strain", stage=1, T_hat=None)
