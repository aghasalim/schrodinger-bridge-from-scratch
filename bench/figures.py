"""Figures, drawn from results/transport.csv only. Nothing is re-measured here."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from sb.bridge_matching import train_bridge_matching
from sb.metrics import sliced_w2
from sb.models import DriftMLP
from sb.ref.toys import sample_pair
from sb.sde import euler_maruyama
from sb.sinkhorn import sinkhorn_plan

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
C = {"sinkhorn": "#762a83", "bridge-matching-sde": "#1a9850",
     "bridge-matching-ode": "#66bd63", "dsb": "#b2182b", "untransported": "#999999"}


def fig_methods(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "transport.csv")
    pairs = sorted(t["pair"].unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.6 * len(pairs), 5.2), squeeze=False)
    for ax, pair in zip(axes[0], pairs):
        sub = t[t["pair"] == pair]
        base = sub[sub["method"] == "untransported"]["w2"].median()
        entries = [
            ("untransported", base, C["untransported"]),
            ("sinkhorn\n(eps 0.03)", sub[sub["method"] == "sinkhorn-eps0.03"]["w2"].median(), C["sinkhorn"]),
            ("DSB\n(best IPF)", sub[(sub["method"] == "dsb")]["w2"].min(), C["dsb"]),
            ("bridge match\n(ODE)", sub[(sub["method"] == "bridge-matching-ode") & (sub["stage"] == 100)]["w2"].median(), C["bridge-matching-ode"]),
            ("bridge match\n(SDE)", sub[(sub["method"] == "bridge-matching-sde") & (sub["stage"] == 100)]["w2"].median(), C["bridge-matching-sde"]),
        ]
        names = [e[0] for e in entries]
        vals = [e[1] for e in entries]
        bars = ax.bar(range(len(entries)), vals, color=[e[2] for e in entries])
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v * 1.05, f"{v:.3f}",
                    ha="center", fontsize=9)
        ax.axhline(base, color="#999999", linestyle=":", linewidth=1.2)
        ax.set_xticks(range(len(entries)))
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_yscale("log")
        ax.set_ylabel("sliced $W_2$ to target (lower is better)")
        ax.set_title(pair)
        ax.grid(alpha=0.3, axis="y", which="both")
    fig.suptitle("Unpaired transport on 2D toys, median of 3 seeds\n"
                 "dotted line is doing nothing at all", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_dsb_ipf(out: Path) -> Path:
    """Does more IPF help? Mostly not, and sometimes it diverges."""
    t = pd.read_csv(RESULTS / "transport.csv")
    d = t[t["method"] == "dsb"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for pair, colour in zip(sorted(d["pair"].unique()), ["#b2182b", "#ef8a62", "#67001f"]):
        sub = d[d["pair"] == pair]
        g = sub.groupby("stage")["w2"]
        med = g.median()
        ax.plot(med.index, med.values, marker="o", color=colour, label=pair, linewidth=1.8)
        n_bad = int(sub["w2"].isna().sum())
        if n_bad:
            ax.scatter([sub[sub["w2"].isna()]["stage"].min()], [med.dropna().iloc[-1]],
                       marker="x", s=110, color=colour, zorder=5)
    ax.set_xlabel("IPF iteration")
    ax.set_ylabel("sliced $W_2$ to target")
    ax.set_title("DSB: more IPF iterations mostly do not help\n"
                 "x marks where a seed went non-finite")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_cost(out: Path) -> Path:
    """Quality against wall clock. The comparison that actually matters."""
    t = pd.read_csv(RESULTS / "transport.csv")
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for method, label, colour, marker in [
            ("sinkhorn-eps0.03", "Sinkhorn (eps 0.03)", C["sinkhorn"], "s"),
            ("dsb", "DSB (per IPF iteration)", C["dsb"], "o"),
            ("bridge-matching-sde", "bridge matching (SDE)", C["bridge-matching-sde"], "^")]:
        sub = t[t["method"] == method]
        if method == "bridge-matching-sde":
            sub = sub[sub["stage"] == 100]
        ax.scatter(sub["wall_s"], sub["w2"], label=label, color=colour,
                   marker=marker, s=42, alpha=0.75)
    ax.set_xlabel("wall clock (s, CPU)")
    ax.set_ylabel("sliced $W_2$ to target")
    ax.set_yscale("log")
    ax.set_title("Quality against compute, all pairs and seeds\n"
                 "down and to the left is better")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_sinkhorn_plan(out: Path) -> Path:
    """The static bridge itself: what entropic regularisation does to the plan."""
    x, y = sample_pair("gaussian->8gaussians", 300, seed=0)
    epss = [1.0, 0.3, 0.1, 0.02]
    fig, axes = plt.subplots(1, len(epss), figsize=(4.0 * len(epss), 4.2))
    for ax, eps in zip(axes, epss):
        P, info = sinkhorn_plan(x, y, eps=eps, iters=600)
        ax.imshow(P.numpy(), cmap="magma", aspect="auto")
        ax.set_title(f"eps = {eps}\n{info['iters']} iters, cost {info['cost']:.3f}", fontsize=10)
        ax.set_xlabel("target index")
        if ax is axes[0]:
            ax.set_ylabel("source index")
    fig.suptitle("Entropic OT plan: small eps concentrates mass onto a near-deterministic map\n"
                 "this is the static Schrodinger bridge, computed exactly", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def _bridge_model(pair, seed=0, steps=4000):
    x0, x1 = sample_pair(pair, 8000, seed=seed)
    torch.manual_seed(seed)
    m = DriftMLP()
    train_bridge_matching(m, x0, x1, sigma=1.0, steps=steps, seed=seed)
    return m, x0, x1


def fig_transport_paths(pair: str, out: Path, seed: int = 0) -> Path:
    m, x0, x1 = _bridge_model(pair, seed)
    probe = x0[:500]
    path = euler_maruyama(m, probe.clone(), steps=80, sigma=1.0, keep_path=True).numpy()
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.scatter(x1[:2500, 0], x1[:2500, 1], s=4, c="#dddddd", label="target $\\pi_1$", zorder=1)
    for i in range(0, 500, 2):
        ax.plot(path[:, i, 0], path[:, i, 1], linewidth=0.4, color="#1a9850", alpha=0.3, zorder=2)
    ax.scatter(path[0, :, 0], path[0, :, 1], s=8, c="#333333", label="source $\\pi_0$", zorder=3)
    ax.scatter(path[-1, :, 0], path[-1, :, 1], s=8, c="#1a9850", label="transported", zorder=4)
    ax.set_title(f"Bridge matching transport, {pair}\n"
                 f"final sliced $W_2$ = {sliced_w2(torch.tensor(path[-1]), x1):.4f}")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def anim_transport(pair: str, out: Path, seed: int = 0, frames: int = 60) -> Path:
    m, x0, x1 = _bridge_model(pair, seed)
    probe = x0[:900]
    path = euler_maruyama(m, probe.clone(), steps=frames, sigma=1.0, keep_path=True).numpy()
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    ax.scatter(x1[:2500, 0], x1[:2500, 1], s=4, c="#dddddd", zorder=1)
    sc = ax.scatter([], [], s=9, c="#1a9850", zorder=3)
    lim = float(np.abs(np.concatenate([x0.numpy(), x1.numpy()])).max()) * 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(alpha=0.22)
    title = ax.set_title("")

    def update(f):
        sc.set_offsets(path[f])
        title.set_text(f"{pair}   bridge matching   t = {f/frames:.2f}")
        return sc, title

    FuncAnimation(fig, update, frames=frames + 1, interval=70).save(out, writer=PillowWriter(fps=14))
    plt.close(fig)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    made = [fig_methods(RESULTS / "methods.png"),
            fig_dsb_ipf(RESULTS / "dsb-ipf.png"),
            fig_cost(RESULTS / "cost-vs-quality.png"),
            fig_sinkhorn_plan(RESULTS / "sinkhorn-plan.png"),
            fig_transport_paths("gaussian->8gaussians", RESULTS / "paths-8gaussians.png"),
            anim_transport("gaussian->8gaussians", RESULTS / "animation-8gaussians.gif")]
    for p in made:
        if p.exists():
            print(f"-> {p.relative_to(REPO)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
