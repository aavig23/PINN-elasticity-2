"""Config values and derived constants match PINN_ELASTICITY_SPEC.md.

Expected numbers are typed in from the spec, independently of the code.
"""

import math

import pytest
import yaml

from conftest import CONFIGS
from pinn_elasticity.config import load_config
from pinn_elasticity.physics.loading import Loading
from pinn_elasticity.physics.material import Material
from pinn_elasticity.physics.scaling import Scales


def test_base_physics_yaml_matches_spec():
    raw = yaml.safe_load((CONFIGS / "base" / "physics.yaml").read_text(encoding="utf-8"))
    assert raw["physics"] == {"L": 1.0, "H": 0.5, "E": 200.0e9, "nu": 0.30, "rho": 7800.0, "P0": 1000.0}
    assert raw["time"] == {"T_hat": 4.0, "t_r_hat": 1.0}  # spec §3.2, §4
    out = raw["output"]
    assert (out["nx"], out["nz"], out["n_snapshots"], out["dpi"]) == (101, 51, 8, 150)  # spec §7.4


@pytest.mark.parametrize("stage", [1, 2])
def test_base_stage_configs_load(stage):
    name = "stage1_static" if stage == 1 else "stage2_dynamic"
    cfg = load_config(CONFIGS / "base" / f"{name}.yaml")
    assert cfg.experiment.stage == stage


def test_default_snapshots_are_decision_c1():
    cfg = load_config(CONFIGS / "base" / "stage2_dynamic.yaml")
    assert cfg.snapshot_times() == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def test_material_constants():
    m = Material(E=200e9, nu=0.30, rho=7800.0)
    assert m.mu == pytest.approx(76.92e9, rel=1e-3)
    assert m.lam_star == pytest.approx(65.93e9, rel=1e-3)
    assert m.lam_3d == pytest.approx(115.38e9, rel=1e-3)
    assert m.lam_star + 2 * m.mu == pytest.approx(219.78e9, rel=1e-3)
    # Exact identities from spec §1.2-1.3.
    assert m.lam_star + 2 * m.mu == pytest.approx(m.E / (1 - m.nu**2), rel=1e-14)
    assert m.lam_star == pytest.approx(2 * m.lam_3d * m.mu / (m.lam_3d + 2 * m.mu), rel=1e-14)


def test_reference_scales():
    s = Scales(L=1.0, H=0.5, P0=1000.0, material=Material(E=200e9, nu=0.30, rho=7800.0))
    assert s.c == pytest.approx(5064, rel=1e-3)
    assert s.t_c == pytest.approx(1.97e-4, rel=5e-3)
    assert s.u_c == pytest.approx(1.0e-8, rel=1e-12)
    assert s.sigma_c == pytest.approx(2.0e3, rel=1e-12)
    assert s.lam_hat == pytest.approx(0.3297, abs=1e-4)
    assert s.mu_hat == pytest.approx(0.3846, abs=1e-4)
    assert s.lam_hat + 2 * s.mu_hat == pytest.approx(1.0989, abs=1e-4)
    assert s.lam_hat == pytest.approx(s.material.lam_star / s.material.E, rel=1e-12)
    assert s.inertia_hat == pytest.approx(1.0, rel=1e-12)  # unit coefficient (spec §3.2)
    assert s.z_hat_max == 0.5
    assert s.time_ms(4.0) == pytest.approx(0.79, rel=5e-3)  # T_hat = 4 -> ~0.79 ms


def test_load_magnitude():
    s = Scales(L=1.0, H=0.5, P0=1000.0, material=Material(E=200e9, nu=0.30, rho=7800.0))
    loading = Loading(s, stage=1, t_r_hat=1.0)
    assert loading.q0 == pytest.approx(2.0e3, rel=1e-12)  # spec §2.5
    assert loading.amplitude_hat == pytest.approx(1.0, rel=1e-12)  # sigma_c = q0


def test_si_round_trips():
    s = Scales(L=1.0, H=0.5, P0=1000.0, material=Material(E=200e9, nu=0.30, rho=7800.0))
    for to_si, to_hat in [
        (s.length_si, s.length_hat),
        (s.time_si, s.time_hat),
        (s.displacement_si, s.displacement_hat),
        (s.stress_si, s.stress_hat),
    ]:
        assert math.isclose(to_hat(to_si(0.37)), 0.37, rel_tol=1e-14)
