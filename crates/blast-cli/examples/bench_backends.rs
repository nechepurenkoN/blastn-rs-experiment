//! Measures complete score-batch latency including GPU transfers/readback.
use blast_core::{
    Result,
    align::{AlignOptions, AlignRequest, GappedAligner},
};
use blast_cpu::ScalarAligner;
use clap::Parser;
use std::{hint::black_box, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long,default_value="cpu",value_parser=["cpu","scalar","simd","gpu","auto"])]
    backend: String,
    #[command(flatten)]
    gpu: blast_cli::GpuTuning,
    #[arg(long, default_value_t = 128)]
    batch_size: usize,
    #[arg(long, default_value_t = 256)]
    length: usize,
    #[arg(long, default_value_t = 10)]
    iterations: usize,
}
fn main() {
    blast_cli::run(run);
}
fn run() -> Result<()> {
    let args = Args::parse();
    if !(32..=10_000).contains(&args.length)
        || args.batch_size == 0
        || args.batch_size > 4096
        || args.iterations == 0
    {
        return Err(blast_core::Error(
            "length 32..10000, batch size 1..4096 and positive iterations required".into(),
        ));
    }
    let mut rng = 41u64;
    let mut data = Vec::new();
    for _ in 0..args.batch_size {
        let q: Vec<_> = (0..args.length)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng >> 32) as u8 % 4
            })
            .collect();
        let mut s = q.clone();
        for i in (19..s.len()).step_by(47) {
            s[i] = (s[i] + 1) % 4;
        }
        s.splice(args.length / 2..args.length / 2, [1, 2, 0]);
        data.push((q, s));
    }
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
    let backend = blast_cli::alignment_backend_with_tuning(&args.backend, args.gpu)?;
    let expected = ScalarAligner.score_batch(&requests, options)?;
    let warmup = backend.score_batch(&requests, options)?;
    if warmup != expected {
        return Err(blast_core::Error("backend disagrees with CPU".into()));
    }
    let start = Instant::now();
    for _ in 0..args.iterations {
        black_box(backend.score_batch(black_box(&requests), options)?);
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("backend\tbatch\tlength\titerations\tseconds\talignments_per_second");
    println!(
        "{}\t{}\t{}\t{}\t{seconds:.6}\t{:.1}",
        backend.name(),
        args.batch_size,
        args.length,
        args.iterations,
        (args.batch_size * args.iterations) as f64 / seconds
    );
    Ok(())
}
