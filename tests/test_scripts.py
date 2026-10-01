"""The Colab helper scripts run as separate processes, as they do in the notebook."""

import subprocess
import sys

import pytest
import torch

from conftest import ROOT


def _run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=600)


def test_check_environment_passes():
    result = _run("scripts/check_environment.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL CHECKS PASSED" in result.stdout
    assert "pinn_elasticity" not in result.stderr


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU failure path")
def test_check_environment_require_gpu_fails_without_gpu():
    result = _run("scripts/check_environment.py", "--require-gpu")
    assert result.returncode == 1
    assert "no CUDA GPU available" in result.stdout


def test_smoke_test_script_passes(tmp_path):
    result = _run("scripts/smoke_test.py", "--device", "cpu", "--out-root", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SMOKE TEST PASSED" in result.stdout
    for stage in (1, 2):
        assert (tmp_path / "smoke_test" / f"stage{stage}" / "summary.json").is_file()
