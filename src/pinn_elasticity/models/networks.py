"""Fully connected network with input normalisation (decision B1).

tanh activation (infinitely differentiable, as spec §5.3 requires at least
twice), Xavier initialisation, inputs mapped affinely from the domain box to
[-1, 1] so the first layer sees O(1) values for every coordinate.
"""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, lower, upper, hidden_layers: int, width: int, out_dim: int = 2):
        super().__init__()
        self.register_buffer("lower", torch.tensor(lower, dtype=torch.float32))
        self.register_buffer("upper", torch.tensor(upper, dtype=torch.float32))
        sizes = [len(lower), *[width] * hidden_layers, out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def normalize(self, X: torch.Tensor) -> torch.Tensor:
        return 2.0 * (X - self.lower) / (self.upper - self.lower) - 1.0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.normalize(X)
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)
