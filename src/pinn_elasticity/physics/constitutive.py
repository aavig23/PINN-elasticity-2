"""Small-strain kinematics and the plane-stress constitutive law (spec §2.1-2.2).

Pure functions of displacement gradients; no autograd happens here. They work
in SI units (pass Material.lam_star, Material.mu) or nondimensionally
(pass Scales.lam_hat, Scales.mu_hat). The argument is named ``lam_star`` on
purpose: plane stress uses lam*, never the 3D lambda (spec §1.3, §5.7).
"""


def strains(u_x, u_z, w_x, w_z) -> dict:
    gamma_xz = u_z + w_x  # engineering shear strain
    return {
        "eps_xx": u_x,
        "eps_zz": w_z,
        "gamma_xz": gamma_xz,
        "eps_xz": 0.5 * gamma_xz,  # tensorial shear strain
    }


def plane_stress_stresses(u_x, u_z, w_x, w_z, *, lam_star, mu):
    """Return (sigma_xx, sigma_zz, sigma_xz)."""
    sigma_xx = (lam_star + 2.0 * mu) * u_x + lam_star * w_z
    sigma_zz = lam_star * u_x + (lam_star + 2.0 * mu) * w_z
    sigma_xz = mu * (u_z + w_x)
    return sigma_xx, sigma_zz, sigma_xz
