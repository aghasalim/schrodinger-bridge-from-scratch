"""Distances between point clouds.

Sliced Wasserstein is the workhorse: exact in each 1D projection, averaged over
random directions. For 2D toys it is cheap and it does not need a feature
network, which FID would.
"""
from __future__ import annotations

import torch


def w2_exact_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Exact 2-Wasserstein in one dimension: sort both and compare."""
    n = min(a.numel(), b.numel())
    return ((a.flatten().sort().values[:n] - b.flatten().sort().values[:n]) ** 2).mean().sqrt().item()


def sliced_w2(a: torch.Tensor, b: torch.Tensor, n_proj: int = 256, seed: int = 0,
              n_quantiles: int = 512) -> float:
    """Sliced 2-Wasserstein, compared on a common quantile grid.

    The obvious implementation sorts both projections and compares element i to
    element i. That is only correct when the two sets are the same size. With
    4000 samples against 8000 it compares all of `a` against the *lowest half*
    of `b`, which reports a large distance between two distributions that are
    in fact the same: a bridge-matching run whose output matched the target's
    mean to 0.01 and its standard deviation to 0.014 was scored at 3.04, and I
    spent a while hunting for a transport bug that was really this.

    Interpolating both onto a shared quantile grid makes the comparison
    size-independent, which is what a distance between distributions has to be.
    """
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n_proj, a.shape[1], generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    pa = (a @ dirs.T).sort(dim=0).values          # (Na, n_proj)
    pb = (b @ dirs.T).sort(dim=0).values          # (Nb, n_proj)

    q = torch.linspace(0.0, 1.0, n_quantiles)
    ia = q * (pa.shape[0] - 1)
    ib = q * (pb.shape[0] - 1)

    def gather(p, idx):
        lo = idx.floor().long().clamp(0, p.shape[0] - 1)
        hi = idx.ceil().long().clamp(0, p.shape[0] - 1)
        w = (idx - lo.float()).unsqueeze(1)
        return p[lo] * (1 - w) + p[hi] * w

    return ((gather(pa, ia) - gather(pb, ib)) ** 2).mean().sqrt().item()


def marginal_error(transported: torch.Tensor, target: torch.Tensor, **kw) -> float:
    """How badly the method missed the target marginal.

    This is the number that matters for a bridge. The whole premise of the
    Schrodinger bridge problem is that the transport hits pi_1 *exactly*, so a
    method that produces a beautiful coupling to the wrong distribution has not
    solved the problem it claimed to.
    """
    return sliced_w2(transported, target, **kw)
