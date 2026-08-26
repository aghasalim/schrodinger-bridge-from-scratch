"""Tests. Each one exists because something went wrong without it."""

import pytest
import torch

from sb.bridge_matching.bridge import BrownianBridge
from sb.metrics import sliced_w2, w2_exact_1d
from sb.ref.toys import PAIRS, sample_pair
from sb.sde import euler_maruyama, ode_sample
from sb.sinkhorn import sinkhorn, sinkhorn_plan, transport_barycentric


# --- metric ----------------------------------------------------------------
def test_sliced_w2_zero_on_identical_sets():
    x = torch.randn(2000, 2)
    assert sliced_w2(x, x) < 1e-6


@pytest.mark.parametrize("na,nb", [(4000, 4000), (4000, 8000), (500, 8000), (8000, 500)])
def test_sliced_w2_is_sample_size_independent(na, nb):
    """The bug that cost the most time in this repo.

    Sorting both projections and comparing element i to element i is only right
    when the sets are the same size. With 4000 against 8000 it compares all of a
    against the lowest half of b, and reports a large distance between two
    samples from the SAME distribution. A working bridge-matching run was scored
    at 3.04 that way while its output matched the target mean to 0.01.
    """
    g = torch.Generator().manual_seed(0)
    pool = torch.randn(8000, 2, generator=g) * 3.0
    same = sliced_w2(pool[:na], pool[:nb])
    shifted = sliced_w2(pool[:na], pool[:nb] + 3.0)

    # Two samples from one distribution never score exactly zero: finite-sample
    # noise is O(1/sqrt(n)) times the spread, which at n=500 and std 3 is a few
    # tenths. So the claim is not "small in absolute terms" but "far smaller
    # than a real difference". A size-dependent metric fails this badly: the
    # broken version scored 4000-vs-8000 samples of one distribution about as
    # far apart as two genuinely different ones.
    assert same < shifted / 4, (
        f"{na} vs {nb} from the same distribution scored {same:.3f}, "
        f"against {shifted:.3f} for a genuine 3.0 shift")


def test_sliced_w2_detects_a_real_shift():
    x = torch.randn(4000, 2)
    assert sliced_w2(x, x + 3.0) > 2.0


def test_w2_exact_1d_matches_known_shift():
    a = torch.linspace(0, 1, 1000)
    assert abs(w2_exact_1d(a, a + 2.0) - 2.0) < 1e-4


# --- sinkhorn --------------------------------------------------------------
def test_sinkhorn_plan_is_a_valid_coupling():
    x, y = sample_pair("gaussian->8gaussians", 200, seed=0)
    P, _ = sinkhorn_plan(x, y, eps=0.1, iters=400)
    assert torch.isfinite(P).all()
    assert abs(P.sum().item() - 1.0) < 1e-4
    # marginals
    assert (P.sum(1) - 1.0 / 200).abs().max() < 1e-4
    assert (P.sum(0) - 1.0 / 200).abs().max() < 1e-4


def test_sinkhorn_survives_small_eps():
    """Log domain is the whole point: the naive version underflows here."""
    x, y = sample_pair("gaussian->8gaussians", 150, seed=1)
    P, _ = sinkhorn_plan(x, y, eps=0.005, iters=300)
    assert torch.isfinite(P).all()


def test_sinkhorn_cost_decreases_with_eps():
    """Less entropic smoothing means a coupling closer to true OT, so lower cost."""
    x, y = sample_pair("gaussian->8gaussians", 200, seed=2)
    costs = [sinkhorn(x, y, eps=e, iters=500)[1]["cost"] for e in (1.0, 0.3, 0.1)]
    assert costs[0] > costs[1] > costs[2]


def test_barycentric_transport_moves_toward_target():
    x, y = sample_pair("gaussian->8gaussians", 800, seed=0)
    xt, _ = transport_barycentric(x, y, eps=0.1)
    assert sliced_w2(xt, y) < sliced_w2(x, y)


# --- bridge ----------------------------------------------------------------
def test_brownian_bridge_hits_both_endpoints():
    x0, x1 = torch.randn(256, 2), torch.randn(256, 2)
    b = BrownianBridge(1.0)
    at0, _ = b.sample(x0, x1, torch.zeros(256))
    at1, _ = b.sample(x0, x1, torch.ones(256))
    assert torch.allclose(at0, x0, atol=1e-5)
    assert torch.allclose(at1, x1, atol=1e-5)


def test_bridge_noise_vanishes_at_endpoints_and_peaks_in_middle():
    x0 = x1 = torch.zeros(4000, 2)
    b = BrownianBridge(1.0)
    spread = [b.sample(x0, x1, torch.full((4000,), t))[0].std().item()
              for t in (0.01, 0.5, 0.99)]
    assert spread[1] > spread[0] and spread[1] > spread[2]
    assert abs(spread[1] - 0.5) < 0.1     # sigma*sqrt(t(1-t)) = 0.5 at t=0.5


def test_exact_bridge_drift_transports_exactly():
    """The ODE and the integrator, checked against the analytic drift.

    dx/dt = (x1 - x)/(1-t) has solution x(t) = x1 + (x0-x1)(1-t), so integrating
    from x0 must land on x1. If this fails the integrator is wrong; if it passes
    and a learned model still misses, the model or the metric is at fault.
    """
    x0 = torch.randn(512, 2)
    x1 = torch.randn(512, 2) * 3

    class Exact(torch.nn.Module):
        def forward(self, x, t):
            return (x1 - x) / (1 - t.view(-1, 1)).clamp_min(1e-4)

    got = ode_sample(Exact(), x0.clone(), steps=200)
    assert (got - x1).abs().max() < 1e-3


# --- sde --------------------------------------------------------------------
def test_zero_drift_zero_noise_is_identity():
    x = torch.randn(64, 2)
    assert torch.allclose(euler_maruyama(lambda a, t: torch.zeros_like(a), x.clone(),
                                         steps=25, sigma=0.0), x)


def test_constant_drift_integrates_exactly():
    c = torch.tensor([0.5, -2.0])
    x = torch.randn(64, 2)
    got = ode_sample(lambda a, t: c.expand_as(a), x.clone(), steps=37)
    assert torch.allclose(got, x + c, atol=1e-5)


def test_backward_uses_positive_dt_with_descending_time():
    """A DSB reverse drift already encodes the reversal.

    `backward=True` must decrement the time label while keeping the position
    update positive. Negating the step as well reverses twice, which sent the
    IPF loop to a forward loss of 1.5e11.
    """
    seen = []

    def drift(x, t):
        seen.append(t[0].item())
        return torch.ones_like(x)

    x = torch.zeros(4, 2)
    got = euler_maruyama(drift, x, steps=4, sigma=0.0, backward=True)
    assert seen[0] > seen[-1], "time label must descend when backward=True"
    assert torch.allclose(got, x + 1.0, atol=1e-6), "position update must stay positive"


def test_noise_scales_with_sigma():
    x = torch.zeros(4000, 2)
    zero = euler_maruyama(lambda a, t: torch.zeros_like(a), x.clone(), steps=50, sigma=0.0)
    noisy = euler_maruyama(lambda a, t: torch.zeros_like(a), x.clone(), steps=50, sigma=1.0)
    assert zero.std() < 1e-6
    assert abs(noisy.std().item() - 1.0) < 0.15     # accumulated variance is sigma^2 * 1


# --- data -------------------------------------------------------------------
@pytest.mark.parametrize("pair", sorted(PAIRS))
def test_pairs_are_finite_and_distinct(pair):
    a, b = sample_pair(pair, 1500, seed=0)
    assert a.shape == b.shape == (1500, 2)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert sliced_w2(a, b) > 0.3, "source and target should not already coincide"


def test_pairs_are_seeded():
    a1, b1 = sample_pair("moons->spiral", 500, seed=4)
    a2, b2 = sample_pair("moons->spiral", 500, seed=4)
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
