import torch

from pinn_elasticity.physics.constitutive import plane_stress_stresses, strains
from pinn_elasticity.physics.material import Material

MAT = Material(E=200e9, nu=0.30, rho=7800.0)


def test_lame_form_equals_engineering_form():
    # Spec §2.2 gives both forms; they must agree for arbitrary gradients.
    g = torch.Generator().manual_seed(0)
    u_x, u_z, w_x, w_z = torch.randn(4, 100, dtype=torch.float64, generator=g) * 1e-6
    sxx, szz, sxz = plane_stress_stresses(u_x, u_z, w_x, w_z, lam_star=MAT.lam_star, mu=MAT.mu)
    E, nu = MAT.E, MAT.nu
    eps = strains(u_x, u_z, w_x, w_z)
    assert torch.allclose(sxx, E / (1 - nu**2) * (eps["eps_xx"] + nu * eps["eps_zz"]), rtol=1e-12)
    assert torch.allclose(szz, E / (1 - nu**2) * (eps["eps_zz"] + nu * eps["eps_xx"]), rtol=1e-12)
    assert torch.allclose(sxz, E / (2 * (1 + nu)) * eps["gamma_xz"], rtol=1e-12)


def test_strain_definitions():
    eps = strains(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0), torch.tensor(4.0))
    assert eps["eps_xx"] == 1.0 and eps["eps_zz"] == 4.0
    assert eps["gamma_xz"] == 5.0 and eps["eps_xz"] == 2.5


def test_plane_stress_guard_uniaxial_state():
    """u = x, w = -nu z is uniaxial tension in plane stress: sxx = E*1, szz = 0, sxz = 0.
    Using the 3D lambda (plane strain) would give szz != 0 (spec §5.7 pitfall)."""
    one, nu = torch.tensor(1.0, dtype=torch.float64), MAT.nu
    zero = torch.tensor(0.0, dtype=torch.float64)
    sxx, szz, sxz = plane_stress_stresses(one, zero, zero, -nu * one, lam_star=MAT.lam_star, mu=MAT.mu)
    assert torch.isclose(sxx, torch.tensor(MAT.E, dtype=torch.float64), rtol=1e-12)
    assert abs(float(szz)) < 1e-12 * MAT.E
    assert float(sxz) == 0.0

    _, szz_3d, _ = plane_stress_stresses(one, zero, zero, -nu * one, lam_star=MAT.lam_3d, mu=MAT.mu)
    assert abs(float(szz_3d)) > 0.1 * MAT.E
