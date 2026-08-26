"""Bridge matching: flow matching with a Brownian bridge interpolant.

Flow matching draws a straight line between x0 and x1 and regresses the velocity
against its slope. Bridge matching replaces the straight line with a Brownian
bridge, which is the reference process conditioned to hit both endpoints:

    x_t = (1-t) x0 + t x1 + sigma * sqrt(t(1-t)) * z,    z ~ N(0, I)

The drift target for that bridge, given the endpoints, is

    b(x_t, t) = (x1 - x_t) / (1 - t)

and regressing against it is simulation free: one network, one loss, no
alternating iterations. That is the whole reason it displaced DSB in practice.

Note the 1/(1-t) blowup as t -> 1. It is real, not an implementation artefact:
the bridge must arrive exactly at x1, so the drift required to correct any
remaining gap diverges. Sampling t away from 1 keeps the regression targets
finite, and `t_max` is that knob.
"""
from __future__ import annotations

import torch


class BrownianBridge:
    """The interpolant and its drift target."""

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def sample(self, x0, x1, t):
        shape = (-1,) + (1,) * (x0.dim() - 1)
        tt = t.view(shape)
        mean = (1 - tt) * x0 + tt * x1
        std = self.sigma * (tt * (1 - tt)).clamp_min(0).sqrt()
        x_t = mean + std * torch.randn_like(mean)
        drift = (x1 - x_t) / (1 - tt).clamp_min(1e-4)
        return x_t, drift


def train_bridge_matching(model, x0_data, x1_data, sigma=1.0, steps=4000,
                          batch=512, lr=2e-3, t_max=0.98, seed=0, log_every=500):
    """Independent coupling, Brownian bridge interpolant, one network."""
    torch.manual_seed(seed)
    bridge = BrownianBridge(sigma)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history = []
    for step in range(steps):
        i = torch.randint(0, x0_data.shape[0], (batch,))
        j = torch.randint(0, x1_data.shape[0], (batch,))
        x0, x1 = x0_data[i], x1_data[j]
        t = torch.rand(batch) * t_max
        x_t, target = bridge.sample(x0, x1, t)
        loss = ((model(x_t, t) - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            history.append({"step": step, "loss": loss.item()})
    return history
