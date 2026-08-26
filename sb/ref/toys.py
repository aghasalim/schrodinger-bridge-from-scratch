"""2D source and target distributions for unpaired transport.

Unlike a generative-model repo, both ends are real distributions here. That is
the whole point of a bridge: neither end has to be Gaussian.
"""
from __future__ import annotations

import math

import torch


def gaussian(n, g, mean=(0.0, 0.0), std=0.5):
    return torch.tensor(mean) + std * torch.randn(n, 2, generator=g)


def eight_gaussians(n, g, scale=4.0, std=0.3):
    c = torch.tensor([(math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8))
                      for i in range(8)]) * scale
    return c[torch.randint(0, 8, (n,), generator=g)] + std * torch.randn(n, 2, generator=g)


def two_moons(n, g, noise=0.1):
    half = n // 2
    a = math.pi * torch.rand(half, generator=g)
    outer = torch.stack([torch.cos(a) * 3, torch.sin(a) * 3], 1)
    b = math.pi * torch.rand(n - half, generator=g)
    inner = torch.stack([1.5 - torch.cos(b) * 3, -torch.sin(b) * 3 + 1.5], 1)
    return torch.cat([outer, inner]) + noise * torch.randn(n, 2, generator=g) * 3


def spiral(n, g, noise=0.08):
    t = torch.rand(n, generator=g) ** 0.5 * 3.5 * math.pi
    r = t * 0.45
    p = torch.stack([r * torch.cos(t), r * torch.sin(t)], 1)
    s = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).view(-1, 1)
    return p * s + noise * torch.randn(n, 2, generator=g) * 3


def circle(n, g, radius=4.0, noise=0.15):
    a = 2 * math.pi * torch.rand(n, generator=g)
    return torch.stack([radius * torch.cos(a), radius * torch.sin(a)], 1) + \
        noise * torch.randn(n, 2, generator=g) * 2


DATASETS = {"gaussian": gaussian, "8gaussians": eight_gaussians,
            "moons": two_moons, "spiral": spiral, "circle": circle}

PAIRS = {
    "gaussian->8gaussians": ("gaussian", "8gaussians"),
    "moons->spiral": ("moons", "spiral"),
    "circle->moons": ("circle", "moons"),
}


def sample_pair(pair: str, n: int, seed: int = 0):
    src, dst = PAIRS[pair]
    g0 = torch.Generator().manual_seed(seed)
    g1 = torch.Generator().manual_seed(seed + 10_000)
    return DATASETS[src](n, g0).float(), DATASETS[dst](n, g1).float()
