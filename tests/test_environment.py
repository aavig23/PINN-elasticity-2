"""M0 gate: the environment and package skeleton are usable."""

import importlib

import pytest
import torch

SUBPACKAGES = ["physics", "models", "training", "postprocess", "experiment", "utils"]


def test_third_party_imports():
    for name in ["torch", "numpy", "matplotlib", "yaml"]:
        importlib.import_module(name)


def test_package_imports_and_has_version():
    pkg = importlib.import_module("pinn_elasticity")
    assert pkg.__version__ == "0.1.0"


@pytest.mark.parametrize("sub", SUBPACKAGES)
def test_subpackages_import(sub):
    importlib.import_module(f"pinn_elasticity.{sub}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_second_derivative_through_autograd(dtype):
    # d2/dx2 sin(x) = -sin(x); needs create_graph=True on the first derivative.
    x = torch.linspace(0.0, 1.0, 11, dtype=dtype, requires_grad=True)
    y = torch.sin(x)
    (dy,) = torch.autograd.grad(y.sum(), x, create_graph=True)
    (d2y,) = torch.autograd.grad(dy.sum(), x)
    tol = 1e-6 if dtype == torch.float32 else 1e-12
    assert torch.allclose(d2y, -torch.sin(x.detach()), atol=tol)
