"""Reference scales and SI <-> nondimensional conversion (spec §3.2).

    c = sqrt(E/rho), t_c = L/c, u_c = P0 L/(E H), sigma_c = E u_c / L
    x_hat = x/L, z_hat = z/L, t_hat = t/t_c, u_hat = u/u_c, sigma_hat = sigma/sigma_c

The nondimensional coefficients are computed from these scales rather than
hard-coded, so lam_hat = lam*/E, mu_hat = mu/E and inertia_hat = 1 follow
from the definitions (and are checked by tests).
"""

import math
from dataclasses import dataclass

from .material import Material


@dataclass(frozen=True)
class Scales:
    L: float  # plate length [m]
    H: float  # plate height [m]
    P0: float  # edge force per unit thickness [N/m]
    material: Material

    @classmethod
    def from_config(cls, cfg) -> "Scales":
        p = cfg.physics
        return cls(L=p.L, H=p.H, P0=p.P0, material=Material.from_config(p))

    # Reference scales -------------------------------------------------------
    @property
    def c(self) -> float:
        """Reference wave speed [m/s]."""
        return math.sqrt(self.material.E / self.material.rho)

    @property
    def t_c(self) -> float:
        """Time for a wave to cross the plate once [s]."""
        return self.L / self.c

    @property
    def u_c(self) -> float:
        """Displacement scale [m]."""
        return self.P0 * self.L / (self.material.E * self.H)

    @property
    def sigma_c(self) -> float:
        """Stress scale [Pa]."""
        return self.material.E * self.u_c / self.L

    # Nondimensional coefficients ---------------------------------------------
    @property
    def lam_hat(self) -> float:
        """lam* in nondimensional stress: sigma_hat contribution per unit grad(u_hat)."""
        return self.material.lam_star * self.u_c / (self.L * self.sigma_c)

    @property
    def mu_hat(self) -> float:
        return self.material.mu * self.u_c / (self.L * self.sigma_c)

    @property
    def inertia_hat(self) -> float:
        """Coefficient of u_hat_tt in the nondimensional equation of motion (= 1)."""
        return self.material.rho * self.u_c * self.L / (self.t_c**2 * self.sigma_c)

    @property
    def z_hat_max(self) -> float:
        """Height of the nondimensional domain, H/L (z_hat = z/L)."""
        return self.H / self.L

    # Conversions -------------------------------------------------------------
    def length_si(self, x_hat):
        return x_hat * self.L

    def length_hat(self, x):
        return x / self.L

    def time_si(self, t_hat):
        return t_hat * self.t_c

    def time_hat(self, t):
        return t / self.t_c

    def time_ms(self, t_hat):
        return t_hat * self.t_c * 1.0e3

    def displacement_si(self, u_hat):
        return u_hat * self.u_c

    def displacement_hat(self, u):
        return u / self.u_c

    def stress_si(self, sigma_hat):
        return sigma_hat * self.sigma_c

    def stress_hat(self, sigma):
        return sigma / self.sigma_c
