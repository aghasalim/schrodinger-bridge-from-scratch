"""Matched-compute comparison of four ways to transport one distribution to another.

  sinkhorn          discrete entropic OT, the static Schrodinger bridge. Ground truth.
  dsb               diffusion Schrodinger bridge by IPF. Two networks, alternating.
  bridge-matching   one network, Brownian-bridge interpolant, simulation free.

The question the repo exists to answer: does full DSB pay for itself against the
simulation-free method that replaced it? Wall clock is recorded for every method
so the comparison is compute-matched rather than iteration-matched.

    .venv/bin/python -m bench.experiment
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from sb.bridge_matching import train_bridge_matching
from sb.dsb import train_dsb
from sb.metrics import sliced_w2
from sb.models import DriftMLP
from sb.ref.toys import PAIRS, sample_pair
from sb.sde import euler_maruyama, ode_sample
from sb.sinkhorn import transport_barycentric

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def run_pair(pair: str, seed: int, n: int, dsb_ipf: int, bm_steps: int) -> list[dict]:
    x0, x1 = sample_pair(pair, n, seed=seed)
    probe = x0[torch.randint(0, n, (n,), generator=torch.Generator().manual_seed(seed + 7))]
    rows = []
    base = sliced_w2(probe, x1, seed=seed)
    rows.append({"pair": pair, "seed": seed, "method": "untransported", "stage": 0,
                 "w2": base, "wall_s": 0.0, "nfe": 0, "params": 0})

    # --- Sinkhorn, several eps ------------------------------------------------
    sub = min(2000, n)
    for eps in (0.5, 0.1, 0.03):
        t0 = time.perf_counter()
        xt, info = transport_barycentric(x0[:sub], x1[:sub], eps=eps, iters=500)
        rows.append({"pair": pair, "seed": seed, "method": f"sinkhorn-eps{eps}",
                     "stage": info["iters"], "w2": sliced_w2(xt, x1, seed=seed),
                     "wall_s": time.perf_counter() - t0, "nfe": 0, "params": 0})

    # --- bridge matching ------------------------------------------------------
    torch.manual_seed(seed)
    m = DriftMLP()
    t0 = time.perf_counter()
    train_bridge_matching(m, x0, x1, sigma=1.0, steps=bm_steps, seed=seed)
    wall = time.perf_counter() - t0
    n_par = sum(p.numel() for p in m.parameters())
    for nfe in (10, 50, 100):
        sde = euler_maruyama(m, probe.clone(), steps=nfe, sigma=1.0)
        ode = ode_sample(m, probe.clone(), steps=nfe)
        rows.append({"pair": pair, "seed": seed, "method": "bridge-matching-sde",
                     "stage": nfe, "w2": sliced_w2(sde, x1, seed=seed),
                     "wall_s": wall, "nfe": nfe, "params": n_par})
        rows.append({"pair": pair, "seed": seed, "method": "bridge-matching-ode",
                     "stage": nfe, "w2": sliced_w2(ode, x1, seed=seed),
                     "wall_s": wall, "nfe": nfe, "params": n_par})

    # --- DSB ------------------------------------------------------------------
    torch.manual_seed(seed)
    fwd, bwd = DriftMLP(), DriftMLP()
    t0 = time.perf_counter()
    hist = train_dsb(fwd, bwd, x0, x1, n_ipf=dsb_ipf, sde_steps=20, inner_steps=800,
                     n_particles=4096, sigma=1.0, seed=seed, verbose=False)
    wall = time.perf_counter() - t0
    n_par = sum(p.numel() for p in fwd.parameters()) * 2
    for e in hist:
        rows.append({"pair": pair, "seed": seed, "method": "dsb", "stage": e["ipf"],
                     "w2": sliced_w2(e["transported"], x1, seed=seed),
                     "wall_s": wall * (e["ipf"] + 1) / dsb_ipf, "nfe": 20, "params": n_par})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=list(PAIRS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--dsb-ipf", type=int, default=10)
    ap.add_argument("--bm-steps", type=int, default=4000)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    started = time.perf_counter()
    rows = []
    for pair in args.pairs:
        for seed in args.seeds:
            print(f"  {pair}  seed {seed}")
            rows += run_pair(pair, seed, args.n, args.dsb_ipf, args.bm_steps)

    out = RESULTS / "transport.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "run-meta.json").write_text(json.dumps({
        "pairs": args.pairs, "seeds": args.seeds, "n": args.n,
        "dsb_ipf": args.dsb_ipf, "bm_steps": args.bm_steps,
        "wall_clock_s": time.perf_counter() - started,
        "torch": torch.__version__, "device": "cpu"}, indent=1))
    print(f"wrote {out.relative_to(REPO)} ({len(rows)} rows) in {time.perf_counter()-started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
