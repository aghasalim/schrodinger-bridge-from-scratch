"""Drift networks. Small, because the toys are 2D and run on a CPU."""
from __future__ import annotations

import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000.0)
                          * torch.arange(half, dtype=t.dtype, device=t.device) / half)
        ang = t.view(-1, 1) * freqs.view(1, -1) * 1000.0
        return self.mlp(torch.cat([ang.sin(), ang.cos()], -1))


class DriftMLP(nn.Module):
    def __init__(self, dim=2, hidden=256, depth=4, t_dim=64):
        super().__init__()
        self.time = TimeEmbedding(t_dim)
        layers = [nn.Linear(dim + t_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.net(torch.cat([x, self.time(t)], -1))
