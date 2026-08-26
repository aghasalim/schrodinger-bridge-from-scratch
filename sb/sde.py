"""Euler-Maruyama, forward and backward.

    dx = b(x,t) dt + sigma dW

The backward direction needs care and is where this repo lost the most time. A
DSB backward drift is fit against

    b_hat(x_{k+1}, t_{k+1}) = b(x_k, t_k) + (x_k - x_{k+1}) / dt

which is arranged so that *stepping with positive dt* moves backward in the
index. The reversal is already inside the target. Integrating it with a negative
dt as well reverses twice and walks away from both marginals: reversing a
Brownian process onto its own start scored W2 4.94 that way against a
do-nothing baseline of 0.56, and the whole IPF loop diverged to a forward loss
of 1.5e11 on top of it. With the sign fixed the same test gives 0.072.

So: `backward=True` decrements the time label while keeping the position update
positive. It is not the same as passing t0=1, t1=0.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def euler_maruyama(drift, x, steps=100, sigma=0.0, keep_path=False, backward=False):
    """Integrate. `backward=True` runs a DSB-style reverse drift (see module doc)."""
    dt = 1.0 / steps
    sq = dt ** 0.5
    path = [x.clone()]
    for i in range(steps):
        t_val = 1.0 - i * dt if backward else i * dt
        t = torch.full((x.shape[0],), t_val, device=x.device)
        x = x + drift(x, t) * dt
        if sigma > 0:
            x = x + sigma * sq * torch.randn_like(x)
        if keep_path:
            path.append(x.clone())
    return torch.stack(path) if keep_path else x


@torch.no_grad()
def ode_sample(drift, x, steps=100, keep_path=False, backward=False):
    """sigma = 0. Deterministic limit, used at sampling time by bridge matching."""
    return euler_maruyama(drift, x, steps=steps, sigma=0.0,
                          keep_path=keep_path, backward=backward)
