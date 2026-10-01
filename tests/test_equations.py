"""Residual operator checks with exactly known fields (code correctness, not accuracy)."""

import math

import pytest
import torch

from pinn_elasticity.physics.equations import evaluate
from pinn_elasticity.physics.material import Material
from pinn_elasticity.physics.scaling import Scales

SCALES = Scales(L=1.0, H=0.5, P0=1000.0, material=Material(E=200e9, nu=0.30, rho=7800.0))
NU = 0.30


class Uniaxial(torch.nn.Module):
    """u_hat = x_hat, w_hat = -nu z_hat: plane-stress uniaxial tension with sigma_hat_xx = 1."""

    def forward(self, X):
        return torch.stack([X[:, 0], -NU * X[:, 1]], dim=1)


class PlaneWave(torch.nn.Module):
    """u_hat = sin(k (x_hat - c t_hat)), w_hat = 0."""

    def __init__(self, k, c):
        super().__init__()
        self.k, self.c = k, c

    def forward(self, X):
        u = torch.sin(self.k * (X[:, 0] - self.c * X[:, 2]))
        return torch.stack([u, 0.0 * X[:, 0]], dim=1)


def _points(n, dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, dim, dtype=torch.float64, generator=g)
    X[:, 1] *= 0.5
    if dim == 3:
        X[:, 2] *= 4.0
    return X


def test_uniaxial_field_is_an_exact_static_solution():
    f = evaluate(Uniaxial(), _points(64, 2), SCALES, stage=1, residual=True)
    assert torch.allclose(f["sxx"], torch.ones(64, dtype=torch.float64), atol=1e-12)
    assert f["szz"].abs().max() < 1e-12  # plane stress: no lateral stress
    assert f["sxz"].abs().max() == 0
    assert f["R_u"].abs().max() < 1e-12 and f["R_w"].abs().max() < 1e-12


def test_plane_wave_satisfies_dynamic_equation_only_at_plate_speed():
    c = math.sqrt((SCALES.lam_hat + 2 * SCALES.mu_hat) / SCALES.inertia_hat)  # = 1/sqrt(1 - nu^2)
    assert c == pytest.approx(1 / math.sqrt(1 - NU**2), rel=1e-12)
    X = _points(64, 3)
    good = evaluate(PlaneWave(k=3.0, c=c), X, SCALES, stage=2, residual=True)
    assert good["R_u"].abs().max() < 1e-10 and good["R_w"].abs().max() < 1e-10

    bad = evaluate(PlaneWave(k=3.0, c=1.1 * c), X, SCALES, stage=2, residual=True)
    assert bad["R_u"].abs().max() > 1e-2  # inertia term really enters the residual


def test_output_shapes_and_keys():
    f1 = evaluate(Uniaxial(), _points(10, 2), SCALES, stage=1, residual=False)
    assert {"u", "w", "u_x", "u_z", "w_x", "w_z", "sxx", "szz", "sxz"} <= set(f1)
    assert "R_u" not in f1 and "u_t" not in f1
    f2 = evaluate(PlaneWave(1.0, 1.0), _points(10, 3), SCALES, stage=2, residual=True)
    assert {"u_t", "w_t", "R_u", "R_w"} <= set(f2)
    assert all(v.shape == (10,) for k, v in f2.items() if k != "X")


def test_wrong_point_dimension_rejected():
    with pytest.raises(ValueError):
        evaluate(Uniaxial(), _points(10, 3), SCALES, stage=1, residual=True)
