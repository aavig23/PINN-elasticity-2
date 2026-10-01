"""Linear isotropic elastic material in plane stress (spec §1.2-1.3)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    E: float  # Young's modulus [Pa]
    nu: float  # Poisson's ratio [-]
    rho: float  # density [kg/m^3]

    @property
    def mu(self) -> float:
        """Shear modulus mu = E / (2(1 + nu))."""
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam_star(self) -> float:
        """Plane-stress effective Lame constant lam* = E nu / (1 - nu^2).

        This is the only lambda used in the constitutive law (spec §1.3).
        """
        return self.E * self.nu / (1.0 - self.nu**2)

    @property
    def lam_3d(self) -> float:
        """3D Lame lambda. Reference only: the spec says it is NOT used directly,
        and using it in place of lam_star silently gives plane strain (spec §5.7)."""
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @classmethod
    def from_config(cls, physics) -> "Material":
        return cls(E=physics.E, nu=physics.nu, rho=physics.rho)
