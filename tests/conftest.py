import shutil
import subprocess
from pathlib import Path

import pytest

from pinn_elasticity.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

# A tiny experiment used by the run/Git tests (never a real experiment).
TEST_EXPERIMENT = "EXP999_guard_test"
TEST_EXPERIMENT_CONFIG = """\
extends: ../../configs/smoke/stage1_smoke.yaml
experiment: {id: EXP999, name: guard_test}
optimizer: {adam_iterations: 6, lbfgs_iterations: 2}
training: {log_every: 2, lbfgs_log_every: 1, checkpoint_every: 4}
"""


def run_git(repo: Path, *args: str) -> str:
    """git in a throwaway test repo, with a fixed identity and no signing."""
    result = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=repo, capture_output=True, text=True, check=True,
    )  # fmt: skip
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path) -> Path:
    """A throwaway git repo with this project's configs, run.py and one committed test experiment."""
    repo = tmp_path / "repo"
    shutil.copytree(CONFIGS, repo / "configs")
    (repo / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "run.py", repo / "scripts" / "run.py")
    shutil.copy(ROOT / ".gitignore", repo / ".gitignore")
    exp = repo / "experiments" / TEST_EXPERIMENT
    exp.mkdir(parents=True)
    (exp / "config.yaml").write_text(TEST_EXPERIMENT_CONFIG, encoding="utf-8")
    run_git(repo, "init", "-q", "-b", "main")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", "init")
    return repo


@pytest.fixture
def make_config():
    """make_config(stage, overrides=None) -> Config built from the smoke config of that stage."""

    def _make(stage: int = 1, overrides: dict | None = None):
        return load_config(CONFIGS / "smoke" / f"stage{stage}_smoke.yaml", overrides)

    return _make
