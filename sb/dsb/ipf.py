"""Diffusion Schrodinger Bridge by Iterative Proportional Fitting.

IPF alternates two half-projections. Starting from a reference process, fit a
backward drift so the reversed process lands on pi_0, then fit a forward drift
so the forward process lands on pi_1, and repeat. Each half-iteration trains a
network on trajectories simulated by the other one, which is why DSB is
expensive: every outer iteration needs a full simulation pass plus a full
training pass, and the two networks can chase each other.

The regression target follows the standard DSB parameterisation: given a
trajectory sampled from the current process, the opposite drift is fit to

    b_hat(x_{k+1}, t_{k+1}) = b(x_k, t_k) + (x_k - x_{k+1}) / dt

which is the discrete time reversal of the transition. This is the form in
De Bortoli et al. 2021, and it is worth writing out because it is the only place
the reversal actually appears.

The honest expectation, stated before running anything: this is more machinery
than bridge matching for the same job, and the comparison in bench/ is designed
to find out whether it pays for itself on these toys.
"""
from __future__ import annotations

import torch

from ..sde import euler_maruyama


@torch.no_grad()
def _simulate(drift, x, steps, sigma, reverse=False):
    """Return the full trajectory and the drift evaluated along it.

    Under no_grad on purpose: the trajectory is *training data* for the next
    half-iteration, not something to backpropagate through. Without this the
    stored drifts keep their graph, and the second inner-loop step dies with
    "Trying to backward through the graph a second time". Simulating with grad
    enabled would also mean backpropagating through 40 SDE steps, which is both
    wrong for IPF and far more memory than the loop needs.
    """
    dt = 1.0 / steps
    sq = dt ** 0.5
    xs, ts, bs = [x.clone()], [], []
    for i in range(steps):
        t_val = 1.0 - i * dt if reverse else i * dt
        t = torch.full((x.shape[0],), t_val, device=x.device)
        b = drift(x, t)
        # Positive dt in BOTH directions. A DSB backward drift already encodes
        # the reversal (see sb/sde.py); negating the step as well reverses twice.
        x = x + b * dt + sigma * sq * torch.randn_like(x)
        xs.append(x.clone())
        ts.append(t)
        bs.append(b)
    return torch.stack(xs), torch.stack(ts), torch.stack(bs)


def _fit_reverse(model, xs, ts, bs, steps_train, batch, lr, sigma, seed, dt_sign=+1.0):
    """Regress the opposite drift against the discrete time reversal.

    `dt_sign` says which way the source trajectory's TIME LABEL moved. A forward
    trajectory has t_k = k*dt so the next state sits at t_k + dt; a reverse one
    has t_k = 1 - k*dt so the next state sits at t_k - dt. Getting this wrong
    trains the forward network against mirrored time labels: the IPF losses fall
    perfectly happily and the transport still ends up worse than doing nothing
    (W2 3.64 against an untransported 2.38), because the network has learned the
    right field indexed by the wrong clock.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dt = 1.0 / (xs.shape[0] - 1)
    n_steps, n_particles = xs.shape[0] - 1, xs.shape[1]
    history = []
    for it in range(steps_train):
        k = torch.randint(0, n_steps, (batch,))
        p = torch.randint(0, n_particles, (batch,))
        x_k = xs[k, p]
        x_k1 = xs[k + 1, p]
        b_k = bs[k, p]
        t_k1 = ts[k, p] + dt_sign * dt
        target = b_k + (x_k - x_k1) / dt
        loss = ((model(x_k1, t_k1.clamp(0.0, 1.0)) - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % max(1, steps_train // 4) == 0:
            history.append({"inner_step": it, "loss": loss.item()})
    return history


class _Zero(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def train_dsb(fwd, bwd, x0_data, x1_data, n_ipf=10, sde_steps=40, inner_steps=800,
              batch=512, lr=2e-3, sigma=1.0, n_particles=2048, seed=0, verbose=True):
    """Alternate IPF. `fwd` transports pi_0 -> pi_1, `bwd` goes the other way."""
    torch.manual_seed(seed)
    history = []
    forward_drift = _Zero()          # first pass uses the reference (Brownian) process

    for k in range(n_ipf):
        # ---- forward simulation from pi_0, fit the backward drift on it
        idx = torch.randint(0, x0_data.shape[0], (n_particles,))
        xs, ts, bs = _simulate(forward_drift, x0_data[idx].clone(), sde_steps, sigma)
        h_b = _fit_reverse(bwd, xs, ts, bs, inner_steps, batch, lr, sigma, seed + k,
                           dt_sign=+1.0)

        # ---- backward simulation from pi_1, fit the forward drift on it
        idx = torch.randint(0, x1_data.shape[0], (n_particles,))
        xs, ts, bs = _simulate(bwd, x1_data[idx].clone(), sde_steps, sigma, reverse=True)
        h_f = _fit_reverse(fwd, xs, ts, bs, inner_steps, batch, lr, sigma, seed + 100 + k,
                           dt_sign=-1.0)

        forward_drift = fwd
        with torch.no_grad():
            probe = x0_data[torch.randint(0, x0_data.shape[0], (2048,))]
            got = euler_maruyama(fwd, probe.clone(), steps=sde_steps, sigma=sigma)
        history.append({"ipf": k, "bwd_loss": h_b[-1]["loss"], "fwd_loss": h_f[-1]["loss"],
                        "transported": got})
        if verbose:
            print(f"      ipf {k:2d}  bwd {h_b[-1]['loss']:.4f}  fwd {h_f[-1]['loss']:.4f}")
    return history
