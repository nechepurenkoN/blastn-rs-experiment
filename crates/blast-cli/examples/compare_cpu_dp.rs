//! Rotating scalar/automatic/forced-SIMD score-only comparison on deterministic workloads.
use blast_core::{
    Error, Result,
    align::{AlignOptions, AlignRequest, GappedAligner},
};
use blast_cpu::{CpuAligner, ScalarAligner, SimdAligner};
use clap::Parser;
use serde_json::json;
use std::{fs::OpenOptions, hint::black_box, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 7)]
    repeats: usize,
    #[arg(long, default_value_t = 10)]
    iterations: usize,
}
fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let i = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[i - 1] + samples[i]) / 2.0
    } else {
        samples[i]
    }
}
fn main() {
    blast_cli::run(run);
}
fn run() -> Result<()> {
    let args = Args::parse();
    if !(3..=100).contains(&args.repeats) || !(1..=1000).contains(&args.iterations) {
        return Err(Error("use 3..100 repeats and 1..1000 iterations".into()));
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.out)?;
    let backends: [&dyn GappedAligner; 3] = [&ScalarAligner, &CpuAligner, &SimdAligner];
    let mut results = Vec::new();
    for length in [64, 256, 1024] {
        for layout in ["uniform", "ragged", "unrelated"] {
            let mut rng = 41u64;
            let mut base = || {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng >> 32) as u8 % 4
            };
            let data: Vec<_> = (0..128)
                .map(|i| {
                    let n = if layout == "ragged" {
                        (length * (1 + i % 4) / 4).max(32)
                    } else {
                        length
                    };
                    let q: Vec<_> = (0..n).map(|_| base()).collect();
                    let mut s = if layout == "unrelated" {
                        (0..n).map(|_| base()).collect()
                    } else {
                        q.clone()
                    };
                    for j in (19..s.len()).step_by(47) {
                        s[j] = (s[j] + 1) % 4;
                    }
                    s.splice(n / 2..n / 2, [1, 2, 0]);
                    s[..11].copy_from_slice(&q[..11]);
                    (q, s)
                })
                .collect();
            let requests: Vec<_> = data
                .iter()
                .map(|(q, s)| AlignRequest {
                    query: q,
                    subject: s,
                    query_anchor: 0,
                    subject_anchor: 0,
                    word_size: 11,
                })
                .collect();
            let options = AlignOptions::default();
            let expected = ScalarAligner.score_batch(&requests, options)?;
            for backend in backends {
                if backend.score_batch(&requests, options)? != expected {
                    return Err(Error(format!(
                        "{} disagrees with scalar: {layout}/{length}",
                        backend.name()
                    )));
                }
            }
            let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
            for round in 0..args.repeats {
                for offset in 0..backends.len() {
                    let index = (round + offset) % backends.len();
                    let start = Instant::now();
                    for _ in 0..args.iterations {
                        black_box(backends[index].score_batch(black_box(&requests), options)?);
                    }
                    samples[index].push(start.elapsed().as_secs_f64());
                }
            }
            let mut medians = [0.0; 3];
            let modes: Vec<_> = backends.into_iter().enumerate().map(|(index, backend)| {
                medians[index] = median(&mut samples[index].clone());
                json!({"backend":backend.name(),"median_seconds":medians[index],"samples_seconds":samples[index]})
            }).collect();
            println!(
                "{layout}/{length}: scalar={:.6}s cpu={:.6}s simd={:.6}s cpu_speedup={:.2}x",
                medians[0],
                medians[1],
                medians[2],
                medians[0] / medians[1]
            );
            results.push(json!({"layout":layout,"length":length,"batch_size":128,"modes":modes}));
        }
    }
    serde_json::to_writer_pretty(file, &json!({
        "arch":std::env::consts::ARCH,"iterations":args.iterations,"repeats":args.repeats,
        "method":"one warmup per backend; rotated order; complete single-thread score_batch including packing/dispatch; no compilation or I/O in timed region",
        "scores_identical":true,"cases":results,
    })).map_err(|e| Error(e.to_string()))?;
    Ok(())
}
