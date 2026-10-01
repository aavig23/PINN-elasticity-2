import torch

from pinn_elasticity.physics.loading import Loading, ramp
from pinn_elasticity.physics.material import Material
from pinn_elasticity.physics.scaling import Scales

SCALES = Scales(L=1.0, H=0.5, P0=1000.0, material=Material(E=200e9, nu=0.30, rho=7800.0))
T_R = 1.0


def test_ramp_values():
    t = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0], dtype=torch.float64)
    f = ramp(t, T_R)
    assert f[0] == 0.0  # compatible with the at-rest initial state (spec §4)
    assert torch.isclose(f[1], torch.tensor(0.5, dtype=torch.float64))
    assert torch.all(f[2:] == 1.0)  # hold after the ramp


def test_ramp_is_monotone_and_continuous():
    t = torch.linspace(0.0, 2.0, 2001, dtype=torch.float64)
    f = ramp(t, T_R)
    assert torch.all(torch.diff(f) >= 0)
    assert abs(float(ramp(torch.tensor(T_R - 1e-9, dtype=torch.float64), T_R)) - 1.0) < 1e-12


def test_ramp_is_c1_slope_zero_at_both_ends():
    t = torch.tensor([0.0, T_R * (1 - 1e-9)], dtype=torch.float64, requires_grad=True)
    (dfdt,) = torch.autograd.grad(ramp(t, T_R).sum(), t)
    assert torch.all(dfdt.abs() < 1e-6)


def test_q_hat_by_stage():
    X1 = torch.rand(7, 2, dtype=torch.float64)
    assert torch.allclose(Loading(SCALES, 1, T_R).q_hat(X1), torch.ones(7, dtype=torch.float64))
    X2 = torch.tensor([[1.0, 0.2, 0.0], [1.0, 0.2, 0.5], [1.0, 0.2, 3.0]], dtype=torch.float64)
    q = Loading(SCALES, 2, T_R).q_hat(X2)
    assert torch.allclose(q, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))
