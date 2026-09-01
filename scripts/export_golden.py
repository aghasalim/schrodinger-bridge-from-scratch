"""Export golden reference values from the Python implementation.

The point is not to publish more numbers. It is to pin down what the Python
Sinkhorn and the Python sliced Wasserstein actually produce on one fixed input,
so implementations in other languages can be required to reproduce them.
Nothing here is a new measurement: the metric reference is checked against a row
that is already in results/transport.csv before it is written.

    python scripts/export_golden.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch

from sb.metrics import sliced_w2
from sb.ref.toys import sample_pair
from sb.sinkhorn.solver import cost_matrix, sinkhorn, transport_barycentric

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "verify" / "golden"

PAIR = "gaussian->8gaussians"
SEED = 0
N = 8000
SUB = 2000        # the cap bench/experiment.py puts on the cost matrix
KERNEL_N = 256    # small enough to commit a full reference for
EPS = (0.5, 0.1, 0.03)


def write_points(path: Path, pts: torch.Tensor) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y"])
        for a, b in pts.tolist():
            w.writerow([f"{a:.10e}", f"{b:.10e}"])


def published_w2() -> float:
    for r in csv.DictReader((REPO / "results" / "transport.csv").open()):
        if r["pair"] == PAIR and int(r["seed"]) == SEED and r["method"] == "sinkhorn-eps0.03":
            return float(r["w2"])
    raise SystemExit("no sinkhorn-eps0.03 row for the golden pair and seed")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    x0, x1 = sample_pair(PAIR, N, seed=SEED)

    # --- the kernel reference: a 256 by 256 problem, committed in full --------
    xk, yk = x0[:KERNEL_N], x1[:KERNEL_N]
    write_points(OUT / "kernel_source.csv", xk)
    write_points(OUT / "kernel_target.csv", yk)
    C = cost_matrix(xk, yk)
    pot = (OUT / "kernel_potentials.csv").open("w", newline="")
    bar = (OUT / "kernel_transported.csv").open("w", newline="")
    sm = (OUT / "kernel_summary.csv").open("w", newline="")
    wp, ws, wb = csv.writer(pot), csv.writer(sm), csv.writer(bar)
    wp.writerow(["eps", "index", "u", "v"])
    wb.writerow(["eps", "index", "x", "y"])
    ws.writerow(["eps", "n", "iters_cap", "iters", "cost", "plan_sum",
                 "row_marginal_max_err", "col_marginal_max_err"])
    for eps in EPS:
        log_P, info = sinkhorn(xk, yk, eps=eps, iters=500)
        P = log_P.exp()
        # The potentials are only defined up to a constant that shifts one and
        # unshifts the other, so a reimplementation is free to land on a
        # different pair. What is pinned here is the plan itself: u is the first
        # column of eps*(log P + C/eps) and v is its first row, both read back
        # off the plan and so free of that gauge.
        fg = eps * (log_P + C / eps)
        u, v = fg[:, 0], fg[0, :]
        for i in range(KERNEL_N):
            wp.writerow([eps, i, f"{u[i]:.12e}", f"{v[i]:.12e}"])
        # The barycentric projection, the step that turns the coupling into a
        # map and the step notes/METHODS.md blames for the circle to moons row.
        Pn = P / P.sum(dim=1, keepdim=True).clamp_min(1e-30)
        for i, (a, b) in enumerate((Pn @ yk).tolist()):
            wb.writerow([eps, i, f"{a:.10e}", f"{b:.10e}"])
        ws.writerow([eps, KERNEL_N, 500, info["iters"], f"{info['cost']:.12e}",
                     f"{P.sum().item():.12e}",
                     f"{(P.sum(1) - 1.0 / KERNEL_N).abs().max().item():.6e}",
                     f"{(P.sum(0) - 1.0 / KERNEL_N).abs().max().item():.6e}"])
    pot.close()
    sm.close()
    bar.close()

    # --- the metric reference: the cloud behind one published sliced W2 -------
    xt, _ = transport_barycentric(x0[:SUB], x1[:SUB], eps=0.03, iters=500)
    got = sliced_w2(xt, x1, seed=SEED)
    want = published_w2()
    if abs(got - want) > 1e-6:
        raise SystemExit(f"export does not reproduce the published w2: {got} vs {want}")
    write_points(OUT / "metric_transported.csv", xt)
    write_points(OUT / "metric_target.csv", x1)
    with (OUT / "metric_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "seed", "method", "n_transported", "n_target",
                    "n_proj", "n_quantiles", "published_w2"])
        w.writerow([PAIR, SEED, "sinkhorn-eps0.03", SUB, N, 256, 512, f"{want:.12e}"])
    print(f"golden export ok, published w2 {want:.6f} reproduced to {abs(got - want):.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
