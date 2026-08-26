# schrodinger-bridge-from-scratch

Entropic optimal transport, iterative proportional fitting, Diffusion Schrödinger Bridges, and the simulation-free bridge-matching methods that largely replaced them — with an honest verdict on when the extra machinery earns its cost.

> **Status: scaffold. Nothing here is built or measured yet.**
> This repo currently holds the project specification, the shared agent conventions,
> and an empty logbook. Every number in the tables below is a `TODO` because no
> experiment has been run. The `prompts/` task specs referenced in the wave table
> are not written yet either.
>
> Nothing in this repo is estimated or taken from a paper. When a table has a number
> in it, that number came from a run in `results/`.

---

## Why

Diffusion models transport a Gaussian to your data. The Schrödinger bridge problem is the general version: given **two arbitrary distributions** and a reference stochastic process, find the process closest in KL to the reference that transports one to the other *exactly*. It's the dynamic form of entropic optimal transport, and standard diffusion is a degenerate special case of it.

This is the most mathematically demanding repo of the eight and the one where the honest conclusion is most likely to be unflattering to its own subject. Full DSB — alternating IPF with two score networks — is expensive and unstable, and the simulation-free bridge-matching methods that came after get most of the benefit for a fraction of the cost. **Establishing that carefully, with matched-compute experiments, is a better project than pretending otherwise**, and it's the kind of result that's genuinely useful to someone deciding what to build.

Do repo 04 first. Bridge matching is flow matching with a Brownian-bridge interpolant instead of a straight line, and the comparison in task 05 is the point of both repos.

## Hardware

- **GPU:** `TODO — python -m scripts.env`
- Toys run on CPU. Image experiments want 12GB+. DSB is iteration-expensive, not memory-expensive — budget wall-clock, not VRAM.

## Results

Unpaired transport, 2D toys — W2 to target vs wall-clock:

| Method | W2 ↓ | Marginal error | Wall-clock | NFE |
|---|---:|---:|---:|---:|
| Sinkhorn (static, discrete) | TODO | TODO | TODO | — |
| DSB (IPF, `n_ipf=20`) | TODO | TODO | TODO | TODO |
| Bridge matching | TODO | TODO | TODO | TODO |
| Flow matching (repo 04) | TODO | TODO | TODO | TODO |

## Waves

```
00 bootstrap + OT metrics                (serial)
   ├─ 01 theory: entropic OT → SB → IPF  ┐
   └─ 02 discrete Sinkhorn               ┘ parallel
        └─ 03 DSB: the hard one          (serial)
             ├─ 04 bridge matching       ┐
             └─ 05 application + comparison ┘ parallel
                  └─ 06 writeup
```

| Task | OWNS | READS |
|---|---|---|
| 00 | `scripts/`, `Makefile`, `sb/metrics/`, `data/` | — |
| 01 | `notes/00-schrodinger.md`, `sb/ref/` | `scripts/` |
| 02 | `sb/sinkhorn/`, `results/sinkhorn/` | `sb/metrics/`, `data/` |
| 03 | `sb/dsb/`, `train/train_dsb.py` | `sb/ref/`, `sb/sinkhorn/` |
| 04 | `sb/bridge_matching/` | `sb/ref/`, `sb/dsb/` |
| 05 | `experiments/` | `sb/dsb/`, `sb/bridge_matching/` |
| 06 | `bench/`, `notes/paper.md`, `README.md` | everything |

See [`CONVENTIONS.md`](CONVENTIONS.md).

## Author

Aghasalim Mustafazada — third-year AI student at Howest, Belgium.

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

MIT — see [LICENSE](LICENSE).
