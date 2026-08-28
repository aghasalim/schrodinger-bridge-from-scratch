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
