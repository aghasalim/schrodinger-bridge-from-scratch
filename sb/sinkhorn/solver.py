"""Discrete entropic optimal transport, in log space.

This is the static Schrodinger bridge. Given cost C and regularisation eps, find
the coupling P minimising <P,C> - eps*H(P) with the two marginals fixed. The
dynamic Schrodinger bridge problem has the same solution at its endpoints, so
this gives the ground truth every learned method in the repo is checked against.

Log-domain throughout. The naive version multiplies exp(-C/eps) directly and
underflows to zero for any eps below about 0.05 on these toys, which shows up as
a plan full of NaNs after the first iteration.
"""
from __future__ import annotations

import torch


def cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared euclidean cost."""
    return torch.cdist(x, y) ** 2


def sinkhorn(x, y, eps: float = 0.1, iters: int = 500, tol: float = 1e-9):
    """Log-domain Sinkhorn. Returns (log_plan, info)."""
    n, m = x.shape[0], y.shape[0]
    C = cost_matrix(x, y)
    log_a = torch.full((n,), -torch.tensor(float(n)).log())
    log_b = torch.full((m,), -torch.tensor(float(m)).log())
    f = torch.zeros(n)
    g = torch.zeros(m)
    K = -C / eps

    err = float("nan")
    for it in range(iters):
        f_prev = f
        # f <- eps * (log a - logsumexp_j (K + g/eps))
        f = eps * (log_a - torch.logsumexp(K + (g / eps).unsqueeze(0), dim=1))
        g = eps * (log_b - torch.logsumexp(K + (f / eps).unsqueeze(1), dim=0))
        err = (f - f_prev).abs().max().item()
        if err < tol:
            break

    log_P = K + (f / eps).unsqueeze(1) + (g / eps).unsqueeze(0)
    info = {"iters": it + 1, "final_update": err, "eps": eps,
            "cost": (log_P.exp() * C).sum().item()}
    return log_P, info


def sinkhorn_plan(x, y, eps=0.1, iters=500):
    log_P, info = sinkhorn(x, y, eps, iters)
    return log_P.exp(), info


def transport_barycentric(x, y, eps=0.1, iters=500):
    """Barycentric projection: map each x_i to the P-weighted mean of the y_j.

    This is a map, not a coupling, so it necessarily blurs. Entropic OT plans are
    spread out by construction, and averaging a spread-out plan pulls mass toward
    the middle. The marginal error this produces is real and is one of the
    numbers worth reporting rather than hiding.
    """
    P, info = sinkhorn_plan(x, y, eps, iters)
    P = P / P.sum(dim=1, keepdim=True).clamp_min(1e-30)
    return P @ y, info
