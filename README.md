# schrodinger-bridge-from-scratch

[![ci](https://github.com/aghasalim/schrodinger-bridge-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/schrodinger-bridge-from-scratch/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![results](https://img.shields.io/badge/results-reproducible-1a9850.svg)](results/)

Entropic optimal transport, diffusion Schrodinger bridges by iterative
proportional fitting, and bridge matching. Built from the papers and compared at
matched compute, on a laptop CPU.

The question this repo exists to answer is whether full DSB pays for itself
against the simulation free method that replaced it. On these toys it does not,
and the gap is large enough that it is worth writing down.

## What the problem is

Diffusion models transport a Gaussian to your data. The Schrodinger bridge
problem is the general version: given two arbitrary distributions and a
reference process, find the process closest in KL to the reference that
transports one to the other exactly. It is the dynamic form of entropic optimal
transport, and standard diffusion is a degenerate special case where one end
happens to be Gaussian.

Three ways to attack it, all implemented here:

**Sinkhorn** solves the static problem directly. Given a cost matrix and a
regularisation eps, find the coupling minimising transport cost minus eps times
entropy. It is exact, it is the ground truth everything else is checked against,
and it does not scale past a few thousand points because it needs the full
cost matrix.

**DSB** solves the dynamic problem by iterative proportional fitting. Two
networks, alternating: fit a backward drift on trajectories from the forward
process, then a forward drift on trajectories from the backward one, and repeat.
Every outer iteration needs a full simulation pass plus a full training pass.

**Bridge matching** replaces the straight interpolant of flow matching with a
Brownian bridge, which is the reference process conditioned to hit both
endpoints. One network, one regression, no alternating iterations, no simulation
during training.

## Results

Sliced Wasserstein-2 to the target, median of 3 seeds, 8000 points per side:

| pair | doing nothing | Sinkhorn (eps 0.03) | DSB (best run) | bridge match (ODE) | bridge match (SDE) |
|---|---:|---:|---:|---:|---:|
| circle to moons | 1.292 | 1.057 | 0.328 | 0.178 | **0.048** |
| gaussian to 8 gaussians | 2.377 | 0.127 | 0.367 | 0.284 | **0.104** |
| moons to spiral | 1.080 | 0.211 | 0.727 | 0.268 | **0.060** |

Every column is the median of 3 seeds except DSB, which is the best of its 30
runs, 10 IPF iterations by 3 seeds. That is the most generous reading of DSB I
can give it and it still loses.

![method comparison](results/methods.png)

Wall clock on the same CPU, per pair:

| method | time | networks | notes |
|---|---:|---:|---|
| Sinkhorn | 2.0 s | 0 | but capped at 2000 points by the cost matrix |
| bridge matching | 8.0 s | 1 | one regression, no simulation while training |
| DSB | 33 s | 2 | 10 IPF iterations, each simulating and training |

Bridge matching is four times cheaper than DSB and closer to the target on every
pair. How much closer depends on which column you read. Against DSB's best run
the ODE column is 1.84x, 1.29x and 2.71x lower in sliced W2, in table row order,
and the SDE column is 6.77x, 3.52x and 12.08x lower.

![cost against quality](results/cost-vs-quality.png)

One thing to read carefully on that figure. Each DSB run was timed once, as a
whole, and that single total is split evenly over its 10 IPF iterations to place
the ten red points. So the height of a DSB point is measured and its position
along the x axis is that even split, which is also why the points fall in
stripes. The 33 s total is the measurement. The experiment now times each IPF
iteration on its own, so a future run will not need the split.

## DSB does not improve with more iterations
Mine does not, and this is the most interesting negative result in the repo.

![DSB across IPF iterations](results/dsb-ipf.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#dsb-does-not-improve-with-more-iterations).
## The static bridge
Sinkhorn is worth looking at directly, because the plan is the object everything else approximates.

![entropic OT plans at four regularisation levels](results/sinkhorn-plan.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#the-static-bridge).
## Transport paths

One bridge matching model, seed 0, integrated forward as an SDE. Each frame is
one Euler Maruyama step out of 60, and the model, the 800 particles and sigma
1.0 are the same in every frame. Grey is the target.

![a gaussian splitting into eight gaussians over 60 SDE steps](results/animation-8gaussians.gif)

The noise spreads the cloud out before the drift sorts it into modes, which is
why the paths are nowhere near straight. All 400 of them at once:

![bridge matching transport paths](results/paths-8gaussians.png)

## What I got wrong
Three bugs, and the order matters because each one hid the next.

Full detail in [notes/METHODS.md](notes/METHODS.md#what-i-got-wrong).
## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python -m pytest tests/ -q
```

```bash
python -m bench.experiment
```

```bash
python -m bench.figures
```

The experiment takes about 7.5 minutes on an M4 CPU and writes
`results/transport.csv`. Figures read that file and never re run an experiment,
so a plot cannot disagree with a number in this README.

## Everything here is checked twice

Every number in this README came out of one implementation. The tables come from
`results/transport.csv`, which comes from `bench/experiment.py`, and the only
thing that had ever checked them was `scripts/check_numbers.py`, which reads the
same file in the same language with the same assumptions. That is not a check.
If the median were taken over the wrong rows, or the Sinkhorn iteration had a
sign error in it, everything downstream would agree, because everything
downstream is the same code.

So the published numbers are now recomputed by implementations that share
nothing with the ones that produced them, and CI fails if any two disagree. An
error would have to be made identically in several languages to survive.

Run them with `./verify/verify.sh`, which skips any toolchain you do not have
and prints how many passed, failed and were skipped.

| language | what it recomputes | from | measured agreement |
|---|---|---|---|
| SQL | all 48 published medians and the wall clock table | `results/transport.csv` | exact at the 3 decimals published |
| C | the log domain Sinkhorn fixed point, in double | `verify/golden/kernel_*.csv` | transport cost within 4.5e-07 relative, potentials within 1.9e-05 absolute |
| Go | file structure: ragged rows, duplicate columns, non finite values, row counts against `run-meta.json` | every results and golden file | 180 rows over 9 runs, 90 DSB rows, 2 non finite, all as documented |
| R | the multiples written in words, and a paired sign test | `results/transport.csv`, `README.md` | all 6 multiples exact as printed, 9 of 9 paired runs, p = 0.0020 |
| Rust | sliced W2 at 32768 projections, and a Monte Carlo interval for the 256 projection estimator | `verify/golden/metric_*.csv` | 0.119069 against the published 0.119723, inside [0.113204, 0.125016] |
| Ruby | the counts asserted in prose | `results/run-meta.json`, `results/transport.csv` | 7 of 7 phrases still true |
| JavaScript | every table cell, in the cell it sits in | `results/transport.csv` | 48 of 48 cells |
| Java | that the plan is a coupling, and its barycentric projection | `verify/golden/kernel_*.csv` | marginals within 3.2e-05 of uniform, map within 3.2e-05, cost within 7.9e-06 relative |

The C, Rust and Java tolerances are not targets. They are what was measured, and
they are as tight as they are because the Python runs in float32 and those three
run in double, so the two sides cannot agree to better than single precision.
The golden files they read are exported by `scripts/export_golden.py`, which
refuses to write anything unless it first reproduces the published sliced W2 for
that pair and seed exactly.

Two of these found something the Python check cannot see. Swap two rows of the
results table above, leaving the row labels where they are, and
`scripts/check_numbers.py` still passes: every number it recomputes is still
somewhere in the document. The JavaScript check resolves each cell by its row
and column heading and reports 10 wrong cells. The R check is the other one, and
it exists because `check_numbers.py` says in its own output that it does not
read claims written in words, which is where the multiples and "four times
cheaper" live.

CI runs `verify/verify.sh`, then corrupts `results/transport.csv`, requires the
run to fail, restores the file and requires it to pass again. A check that
cannot fail is not checking anything, and this is how that stays true.

## Layout

```
sb/metrics/      sliced W2 on a shared quantile grid, marginal error
sb/ref/toys.py   2D source and target pairs
sb/sinkhorn/     log domain entropic OT, the static bridge
sb/dsb/          IPF with two drift networks
sb/bridge_matching/  Brownian bridge interpolant, simulation free
sb/sde.py        Euler Maruyama, forward and backward
sb/models.py     drift MLP with sinusoidal time embedding
bench/           the experiment and the figures
verify/          the published numbers recomputed in eight other languages
tests/           22 tests
```

## Sources

- **Cuturi. Sinkhorn Distances: Lightspeed Computation of Optimal Transport. NeurIPS 2013.** [arXiv:1306.0895](https://arxiv.org/abs/1306.0895) The entropic regularisation and the fixed point iteration.
- **Peyré, Cuturi. Computational Optimal Transport. FnT ML 2019.** [arXiv:1803.00567](https://arxiv.org/abs/1803.00567) The log domain stabilisation used here, and the barycentric projection caveat.
- **De Bortoli, Thornton, Heng, Doucet. Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling. NeurIPS 2021.** [arXiv:2106.01357](https://arxiv.org/abs/2106.01357) DSB and the IPF scheme. The discrete time reversal target is theirs.
- **Shi, De Bortoli, Campbell, Doucet. Diffusion Schrödinger Bridge Matching. NeurIPS 2023.** [arXiv:2303.16852](https://arxiv.org/abs/2303.16852) Bridge matching, and why the alternating scheme can be replaced.
- **Liu, Wu, Ye, Zhu. I2SB: Image-to-Image Schrödinger Bridge. ICML 2023.** [arXiv:2302.05872](https://arxiv.org/abs/2302.05872) The tractable bridge construction that makes the simulation free version work.
- **Léonard. A survey of the Schrödinger problem and some of its connections with optimal transport. 2013.** [arXiv:1308.0215](https://arxiv.org/abs/1308.0215) The link between the Schrodinger problem and entropic OT.
- **Lipman et al. Flow Matching for Generative Modeling. ICLR 2023.** [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) Bridge matching is this with a different interpolant. See [rectified-flow-from-scratch](https://github.com/aghasalim/rectified-flow-from-scratch).

## Methodology

The rules this follows are in [`METHODOLOGY.md`](METHODOLOGY.md). Rule 8, no number
that did not come from a measurement, and rule 14, negative results stay in, are
why the DSB section reads the way it does.

## Author

Aghasalim Mustafazada, third year AI student at Howest, Belgium.

<p align="center">
  <a href="https://github.com/aghasalim">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="github"></a>
  <a href="https://www.kaggle.com/aghasalimmustafazada">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="kaggle"></a>
  <a href="https://linkedin.com/in/mustafazada">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="linkedin"></a>
  <a href="https://orcid.org/0009-0001-8746-4582">
    <img src="https://img.shields.io/badge/ORCID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="orcid"></a>
</p>

## License

MIT, see [LICENSE](LICENSE).
