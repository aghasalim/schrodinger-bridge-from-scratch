// Sliced Wasserstein-2, reimplemented, with a Monte Carlo error bar on the
// published number.
//
// The sliced W2 in sb/metrics/distances.py averages over 256 random
// projections, so every number in the results table is a Monte Carlo estimate
// and none of them was ever given an error bar. Two questions follow, and
// neither can be answered by the Python because the Python is the thing under
// question: does an independent implementation of the metric land in the same
// place, and is 256 projections enough for the published digits to mean
// anything.
//
// This recomputes the metric from verify/golden/metric_*.csv with far more
// projections than the experiment could afford, then resamples projections to
// build the sampling distribution of the published 256 projection estimator and
// requires the published value to fall inside it. No crates: the PRNG is a
// xorshift written out below, so the only thing shared with the Python is the
// input file.

use std::env;
use std::fs;
use std::process::exit;

const N_PROJ: usize = 32768;
const N_BOOT: usize = 20000;

struct Xorshift(u64);
impl Xorshift {
    fn new(seed: u64) -> Self {
        Xorshift(seed.wrapping_mul(0x9E3779B97F4A7C15).max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        // 53 bits into [0,1)
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        // Box Muller. One of the two values is discarded, which costs nothing
        // here and keeps the call site simple.
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn split_csv(line: &str) -> Vec<&str> {
    line.trim_end_matches(['\r', '\n']).split(',').collect()
}

fn column_of(header: &str, want: &str) -> usize {
    split_csv(header)
        .iter()
        .position(|c| *c == want)
        .unwrap_or_else(|| {
            eprintln!("slicedw2: no column named {want}");
            exit(1);
        })
}

fn read_points(path: &str) -> Vec<[f64; 2]> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("slicedw2: cannot read {path}: {e}");
        exit(1);
    });
    let mut lines = text.lines();
    let header = lines.next().unwrap_or_else(|| {
        eprintln!("slicedw2: {path} is empty");
        exit(1);
    });
    let (cx, cy) = (column_of(header, "x"), column_of(header, "y"));
    let mut out = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f = split_csv(line);
        let a: f64 = f[cx].parse().unwrap_or_else(|_| {
            eprintln!("slicedw2: {path}: not a number: {}", f[cx]);
            exit(1);
        });
        let b: f64 = f[cy].parse().unwrap_or_else(|_| {
            eprintln!("slicedw2: {path}: not a number: {}", f[cy]);
            exit(1);
        });
        if !a.is_finite() || !b.is_finite() {
            eprintln!("slicedw2: {path}: non finite point");
            exit(1);
        }
        out.push([a, b]);
    }
    out
}

// The shared quantile grid from sb/metrics/distances.py: n_quantiles points
// evenly spaced in [0,1], each read off the sorted projection by linear
// interpolation between the two neighbouring order statistics.
fn on_grid(sorted: &[f64], n_q: usize, out: &mut Vec<f64>) {
    out.clear();
    let last = (sorted.len() - 1) as f64;
    for k in 0..n_q {
        let q = k as f64 / (n_q - 1) as f64;
        let idx = q * last;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        let w = idx - lo as f64;
        out.push(sorted[lo] * (1.0 - w) + sorted[hi] * w);
    }
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let a = read_points(&format!("{root}/verify/golden/metric_transported.csv"));
    let b = read_points(&format!("{root}/verify/golden/metric_target.csv"));

    let summary = fs::read_to_string(format!("{root}/verify/golden/metric_summary.csv"))
        .unwrap_or_else(|e| {
            eprintln!("slicedw2: cannot read metric_summary.csv: {e}");
            exit(1);
        });
    let mut sl = summary.lines();
    let header = sl.next().unwrap();
    let c_w2 = column_of(header, "published_w2");
    let c_nt = column_of(header, "n_transported");
    let c_ng = column_of(header, "n_target");
    let c_np = column_of(header, "n_proj");
    let c_nq = column_of(header, "n_quantiles");
    let row = sl.next().unwrap_or_else(|| {
        eprintln!("slicedw2: metric_summary.csv has no data row");
        exit(1);
    });
    let f = split_csv(row);
    let published: f64 = f[c_w2].parse().unwrap();
    let n_t: usize = f[c_nt].parse().unwrap();
    let n_g: usize = f[c_ng].parse().unwrap();
    let n_proj_pub: usize = f[c_np].parse().unwrap();
    let n_q: usize = f[c_nq].parse().unwrap();

    if a.len() != n_t || b.len() != n_g {
        eprintln!(
            "slicedw2: point counts disagree with the summary: {} and {}, expected {} and {}",
            a.len(),
            b.len(),
            n_t,
            n_g
        );
        exit(1);
    }
    println!("  {} transported points, {} target points", a.len(), b.len());

    // Per projection mean squared quantile gap. The published metric is the
    // square root of the mean of these over the projections.
    let mut rng = Xorshift::new(20240917);
    let mut per_proj = Vec::with_capacity(N_PROJ);
    let (mut pa, mut pb) = (vec![0.0; a.len()], vec![0.0; b.len()]);
    let (mut ga, mut gb) = (Vec::new(), Vec::new());
    for _ in 0..N_PROJ {
        let (mut d0, mut d1) = (rng.normal(), rng.normal());
        let norm = (d0 * d0 + d1 * d1).sqrt();
        if norm < 1e-12 {
            continue;
        }
        d0 /= norm;
        d1 /= norm;
        for (i, p) in a.iter().enumerate() {
            pa[i] = p[0] * d0 + p[1] * d1;
        }
        for (i, p) in b.iter().enumerate() {
            pb[i] = p[0] * d0 + p[1] * d1;
        }
        pa.sort_by(|x, y| x.partial_cmp(y).unwrap());
        pb.sort_by(|x, y| x.partial_cmp(y).unwrap());
        on_grid(&pa, n_q, &mut ga);
        on_grid(&pb, n_q, &mut gb);
        let mut s = 0.0;
        for k in 0..n_q {
            let d = ga[k] - gb[k];
            s += d * d;
        }
        per_proj.push(s / n_q as f64);
    }

    let mean: f64 = per_proj.iter().sum::<f64>() / per_proj.len() as f64;
    let estimate = mean.sqrt();

    // Sampling distribution of the published estimator: resample n_proj_pub
    // projections with replacement and take the same square root of the mean.
    let mut boot = Vec::with_capacity(N_BOOT);
    for _ in 0..N_BOOT {
        let mut s = 0.0;
        for _ in 0..n_proj_pub {
            s += per_proj[rng.below(per_proj.len())];
        }
        boot.push((s / n_proj_pub as f64).sqrt());
    }
    boot.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let lo = boot[(0.005 * N_BOOT as f64) as usize];
    let hi = boot[(0.995 * N_BOOT as f64) as usize];

    println!(
        "  {N_PROJ} projections give {estimate:.6}, published {n_proj_pub} projection value {published:.6}"
    );
    println!(
        "  {N_BOOT} resamples of {n_proj_pub} projections: 99% interval [{lo:.6}, {hi:.6}]"
    );
    println!("  gap to the published value {:.2e}", (estimate - published).abs());

    if published < lo || published > hi {
        println!("  FAIL: the published value is outside the interval its own estimator produces");
        exit(1);
    }
    println!("  Rust reproduces the published sliced W2 inside the Monte Carlo interval");
}
