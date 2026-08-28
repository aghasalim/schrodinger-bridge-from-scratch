"""Figures. Every quoted number is read from results/transport.csv, never remeasured.

Three of these are pictures of a computation rather than of the CSV: the
Sinkhorn plan, the transport paths and the animation. They re-run the repo's own
code at a fixed seed and they are deterministic, but they still quote nothing of
their own. The one number that appears on the paths figure comes from the CSV.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from bench.style import PALETTE, titled
from sb.bridge_matching import train_bridge_matching
from sb.models import DriftMLP
from sb.ref.toys import sample_pair
from sb.sde import euler_maruyama
from sb.sinkhorn import sinkhorn_plan

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

BLUE, RED, GREEN, ORANGE, PURPLE, SLATE = PALETTE
GREY = "#9a9a9a"
# Colour by method, held to the same meaning in every figure. Green is bridge
# matching, red is DSB, purple is Sinkhorn, grey is doing nothing at all.
C = {"untransported": GREY, "sinkhorn-eps0.03": PURPLE, "dsb": RED,
     "bridge-matching-ode": "#7cc47f", "bridge-matching-sde": GREEN}
W2LAB = "sliced $W_2$ to the target (data units, lower is better)"
PAIR = {"gaussian->8gaussians": "gaussian to 8 gaussians",
        "moons->spiral": "moons to spiral", "circle->moons": "circle to moons"}



def _shrink_gif(path: Path, colours: int = 64) -> None:
    """Rewrite the GIF on one shared palette.

    PillowWriter gives every frame its own full palette, which is most of the file
    size and is wasted here: consecutive frames differ only slightly, so one palette
    taken from a middle frame covers all of them and lets the encoder store just the
    changes. Colour count is high enough that the antialiased text does not band.
    """
    from PIL import Image

    source = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(source.convert("RGB"))
            durations.append(source.info.get("duration", 62))
            source.seek(source.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(colours, method=Image.Quantize.MEDIANCUT)
    quantised = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    # No disposal method: leaving it unset lets the encoder store only the region
    # that changed between frames. Setting disposal=2 forces a full redraw each
    # frame and made the file larger than the one it replaced.
    quantised[0].save(path, save_all=True, append_images=quantised[1:], loop=0,
                      duration=durations, optimize=True)

def _table(t: pd.DataFrame) -> pd.DataFrame:
    """The five headline numbers per pair, exactly as the README table quotes them."""
    rows = []
    for pair in sorted(t["pair"].unique()):
        s = t[t["pair"] == pair]
        rows.append({
            "pair": pair,
            "untransported": s[s["method"] == "untransported"]["w2"].median(),
            "sinkhorn-eps0.03": s[s["method"] == "sinkhorn-eps0.03"]["w2"].median(),
            # The DSB entry is its best single run, not a median. See the subtitle.
            "dsb": s[s["method"] == "dsb"]["w2"].min(),
            "bridge-matching-ode": s[(s["method"] == "bridge-matching-ode")
                                     & (s["stage"] == 100)]["w2"].median(),
            "bridge-matching-sde": s[(s["method"] == "bridge-matching-sde")
                                     & (s["stage"] == 100)]["w2"].median(),
        })
    return pd.DataFrame(rows).set_index("pair")


def fig_methods(out: Path) -> Path:
    tab = _table(pd.read_csv(RESULTS / "transport.csv"))
    methods = [("untransported", "doing nothing"),
               ("sinkhorn-eps0.03", "Sinkhorn, eps 0.03"),
               ("dsb", "DSB, best run"),
               ("bridge-matching-ode", "bridge matching, ODE"),
               ("bridge-matching-sde", "bridge matching, SDE")]
    h = 0.15
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for k, (key, label) in enumerate(methods):
        offs = k - (len(methods) - 1) / 2
        ys = np.arange(len(tab)) + offs * h
        vals = tab[key].to_numpy()
        ax.barh(ys, vals, height=h * 0.92, color=C[key], label=label, zorder=3)
        for y, v in zip(ys, vals):
            ax.text(v + 0.035, y, f"{v:.3f}", va="center", ha="left",
                    fontsize=8.6, color="#333333", zorder=4)
    ax.set_yticks(np.arange(len(tab)))
    ax.set_yticklabels([PAIR[p] for p in tab.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 2.72)
    ax.set_xlabel(W2LAB)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              columnspacing=1.4, handlelength=1.3, fontsize=9.0)
    titled(ax, "Bridge matching wins on all three pairs",
           "8000 points a side, median of 3 seeds, 100 sampling steps. The DSB bar is its "
           "best of 30 runs, 10 IPF iterations by 3 seeds, so it flatters DSB.")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_dsb_ipf(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "transport.csv")
    d = t[t["method"] == "dsb"]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for pair, colour in zip(sorted(d["pair"].unique()), [BLUE, ORANGE, PURPLE]):
        sub = d[d["pair"] == pair]
        # Three seeds is too few for a median alone to be honest, so the raw
        # seeds sit faintly behind the line rather than being averaged away.
        ax.scatter(sub["stage"], sub["w2"], s=16, color=colour, alpha=0.3, zorder=2)
        med = sub.groupby("stage")["w2"].median()
        ax.plot(med.index, med.to_numpy(), marker="o", color=colour,
                label=PAIR[pair], zorder=3)
        bad = sub[sub["w2"].isna()]["stage"]
        if len(bad):
            ax.scatter([bad.min()], [med.loc[bad.min()]], marker="x", s=130,
                       linewidth=2.4, color=colour, zorder=5)
    ax.set_xlabel("IPF iteration (each one is a full simulate and train pass)")
    ax.set_ylabel(W2LAB)
    ax.set_xticks(range(10))
    ax.set_ylim(0.25, 1.48)
    ax.legend(loc="upper left", ncol=3)
    titled(ax, "More IPF iterations do not fix my DSB",
           "line is the median of 3 seeds, faint dots are the seeds themselves. The cross is "
           "where one seed went non-finite and stayed that way, so the line past it is 2 seeds.")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_cost(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "transport.csv")
    # The DSB run in the committed CSV was timed once and that one total was split
    # evenly over its IPF iterations, which is why the red points sit in stripes.
    # Read that off the data instead of assuming it: a run whose iterations timed
    # themselves drops the caveat here rather than keeping a stale one.
    d = t[t["method"] == "dsb"]
    n_ipf = int(d["stage"].max()) + 1
    total = d.groupby(["pair", "seed"])["wall_s"].transform("max")
    split = bool(np.allclose(d["wall_s"], total * (d["stage"] + 1) / n_ipf))
    xlab = "wall clock (seconds, M4 CPU)"
    dsb_lab = "DSB, one point per IPF iteration"
    if split:
        xlab += f". DSB is one measured total per run, split evenly over {n_ipf} iterations"
        dsb_lab += "\nx is that even split, not a timing per iteration"
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    # Labelled in place instead of with a legend, so nothing sits on the data.
    for method, label, colour, marker, xy, off in [
            ("sinkhorn-eps0.03", "Sinkhorn\ncapped at 2000 points", PURPLE, "s", (2.0, 0.30), (10, 8)),
            ("dsb", dsb_lab, RED, "o", (11.0, 1.62), (0, 0)),
            ("bridge-matching-sde", "bridge matching", GREEN, "^", (8.2, 0.045), (12, -4))]:
        sub = t[t["method"] == method]
        if method == "bridge-matching-sde":
            sub = sub[sub["stage"] == 100]
        ax.scatter(sub["wall_s"], sub["w2"], color=colour, marker=marker, s=40,
                   alpha=0.7, linewidth=0, zorder=3)
        ax.annotate(label, xy=xy, xytext=off, textcoords="offset points",
                    color=colour, fontsize=10, fontweight="semibold",
                    ha="left", va="center", zorder=4)
    ax.set_xlabel(xlab)
    ax.set_ylabel(W2LAB)
    ax.set_yscale("log")
    ax.set_xlim(0, 37)
    ax.set_ylim(0.035, 2.3)
    titled(ax, "The cheap method is also the accurate one",
           "every run, all 3 pairs and 3 seeds, at 100 sampling steps. Down and to the left is better.")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_sinkhorn_plan(out: Path, n: int = 300, seed: int = 0) -> Path:
    """The static bridge. Rows and columns are sorted by angle so shape is visible."""
    x, y = sample_pair("gaussian->8gaussians", n, seed=seed)
    oi = np.argsort(np.arctan2(x[:, 1].numpy(), x[:, 0].numpy()))
    oj = np.argsort(np.arctan2(y[:, 1].numpy(), y[:, 0].numpy()))
    epss = [1.0, 0.3, 0.1, 0.02]
    fig, axes = plt.subplots(1, len(epss), figsize=(12.6, 3.9))
    norm = LogNorm(vmin=0.05, vmax=n)
    for ax, eps in zip(axes, epss):
        P, _ = sinkhorn_plan(x, y, eps=eps, iters=600)
        # In multiples of the uniform plan: 1 is "no preference", n is a
        # deterministic assignment of that source point to that target point.
        Q = P.numpy()[np.ix_(oi, oj)] * n * n
        im = ax.imshow(np.clip(Q, 0.05, None), cmap="magma", norm=norm,
                       aspect="equal", interpolation="nearest")
        ax.text(0.035, 0.965, f"eps = {eps}", transform=ax.transAxes, color="white",
                fontsize=10.5, fontweight="semibold", va="top", ha="left")
        ax.set_xlabel("target point, sorted by angle")
        ax.grid(False)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].set_ylabel("source point, sorted by angle")
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.015)
    cb.set_label("coupling mass, multiples of the uniform plan", fontsize=9.3)
    cb.outline.set_visible(False)
    titled(axes[0], "Shrinking eps collapses the coupling onto a map",
           "the same 300 by 300 entropic plan at four regularisation levels. Sorting both axes by "
           "angle turns the eight target modes into the eight blocks.")
    fig.savefig(out)
    plt.close(fig)
    return out


@lru_cache(maxsize=2)
def _bridge_model(pair: str, seed: int = 0, steps: int = 4000):
    """One trained bridge matching model, shared by the paths figure and the animation."""
    x0, x1 = sample_pair(pair, 8000, seed=seed)
    torch.manual_seed(seed)
    m = DriftMLP()
    train_bridge_matching(m, x0, x1, sigma=1.0, steps=steps, seed=seed)
    return m, x0, x1


def _rollout(pair: str, n_probe: int, steps: int, seed: int = 0):
    """Simulate n_probe source points forward under the learned drift. Deterministic."""
    m, x0, x1 = _bridge_model(pair, seed)
    torch.manual_seed(seed + 1)
    path = euler_maruyama(m, x0[:n_probe].clone(), steps=steps, sigma=1.0,
                          keep_path=True).numpy()
    return path, x1.numpy()


def fig_transport_paths(pair: str, out: Path, seed: int = 0) -> Path:
    measured = _table(pd.read_csv(RESULTS / "transport.csv")).loc[pair, "bridge-matching-sde"]
    path, x1 = _rollout(pair, 400, 80, seed)
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.scatter(x1[:2500, 0], x1[:2500, 1], s=5, c="#dcdcdc", label="target $\\pi_1$", zorder=1)
    seg = np.stack([path[:-1], path[1:]], axis=2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
    ax.add_collection(LineCollection(seg, colors=GREEN, linewidths=0.35, alpha=0.16, zorder=2))
    ax.scatter(path[0, :, 0], path[0, :, 1], s=9, c=SLATE, label="source $\\pi_0$", zorder=3)
    ax.scatter(path[-1, :, 0], path[-1, :, 1], s=9, c=GREEN, label="transported", zorder=4)
    ax.set_xlabel("$x_1$"), ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.set_xlim(-6.2, 6.2), ax.set_ylim(-6.2, 6.2)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    titled(ax, "Every mode gets filled, and no path is straight",
           f"400 SDE paths, sigma 1.0, seed {seed}, 80 steps. Measured on the full "
           f"8000 point benchmark this transport scores {measured:.3f} sliced $W_2$.")
    fig.savefig(out)
    plt.close(fig)
    return out


def anim_transport(pair: str, out: Path, seed: int = 0, steps: int = 60,
                   fps: int = 15, hold: int = 14) -> Path:
    """The transport unfolding in time. Same model and same seed as the paths figure."""
    path, x1 = _rollout(pair, 800, steps, seed)
    trail = 6
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.scatter(x1[:2500, 0], x1[:2500, 1], s=5, c="#dcdcdc", zorder=1)
    tails = LineCollection([], colors=GREEN, linewidths=0.5, alpha=0.22, zorder=2)
    ax.add_collection(tails)
    dots = ax.scatter(path[0, :, 0], path[0, :, 1], s=8, c=GREEN, zorder=3)
    ax.set_xlim(-6.2, 6.2), ax.set_ylim(-6.2, 6.2)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$"), ax.set_ylabel("$x_2$")
    titled(ax, "One Gaussian splitting into eight",
           f"bridge matching, {steps} Euler Maruyama steps, sigma 1.0, seed {seed}")
    clock = ax.text(0.975, 0.025, "", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=11, color=SLATE, fontweight="semibold")
    bar = ax.axhline(-6.2, xmin=0.0, xmax=0.0, color=GREEN, linewidth=3.5, zorder=6)

    def update(f):
        f = min(f, steps)
        dots.set_offsets(path[f])
        lo = max(0, f - trail)
        seg = np.stack([path[lo:f], path[lo + 1:f + 1]], axis=2)
        tails.set_segments(seg.transpose(1, 0, 2, 3).reshape(-1, 2, 2) if f > lo else [])
        clock.set_text(f"t = {f / steps:.2f}")
        bar.set_xdata([0.0, f / steps])
        return dots, tails, clock, bar

    anim = FuncAnimation(fig, update, frames=steps + 1 + hold, interval=1000 // fps)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=72)
    _shrink_gif(out)
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
