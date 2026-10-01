import torch

from pinn_elasticity.utils.autodiff import gradient


def test_first_and_second_derivatives_match_closed_forms():
    g = torch.Generator().manual_seed(1)
    X = torch.rand(50, 2, dtype=torch.float64, generator=g).requires_grad_(True)
    x, z = X[:, 0], X[:, 1]
    y = torch.sin(2 * x) * torch.cos(3 * z)

    d = gradient(y, X)
    assert torch.allclose(d[:, 0], 2 * torch.cos(2 * x) * torch.cos(3 * z), atol=1e-12)
    assert torch.allclose(d[:, 1], -3 * torch.sin(2 * x) * torch.sin(3 * z), atol=1e-12)

    y_xx = gradient(d[:, 0], X)[:, 0]
    y_xz = gradient(d[:, 0], X)[:, 1]
    assert torch.allclose(y_xx, -4 * y, atol=1e-12)
    assert torch.allclose(y_xz, -6 * torch.cos(2 * x) * torch.sin(3 * z), atol=1e-12)
    assert y_xx.abs().max() > 0.1  # a missing create_graph would make this zero or fail


def test_output_independent_of_input_gives_zeros():
    X = torch.rand(5, 3, dtype=torch.float64).requires_grad_(True)
    assert torch.equal(gradient(torch.ones(5, dtype=torch.float64), X), torch.zeros_like(X))
