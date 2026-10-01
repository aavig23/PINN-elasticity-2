from pathlib import Path

import pytest

from pinn_elasticity.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture
def make_config():
    """make_config(stage, overrides=None) -> Config built from the smoke config of that stage."""

    def _make(stage: int = 1, overrides: dict | None = None):
        return load_config(CONFIGS / "smoke" / f"stage{stage}_smoke.yaml", overrides)

    return _make
