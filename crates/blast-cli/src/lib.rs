use blast_core::{Error, Result};
use std::{
    ffi::OsString,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};
pub fn input(path: &Path) -> Result<Box<dyn BufRead>> {
    if path == Path::new("-") {
        Ok(Box::new(BufReader::new(std::io::stdin())))
    } else {
        Ok(Box::new(BufReader::new(File::open(path)?)))
    }
}
pub fn output(path: Option<&Path>) -> Result<Box<dyn Write>> {
    match path {
        None => Ok(Box::new(BufWriter::new(std::io::stdout()))),
        Some(p) if p == Path::new("-") => Ok(Box::new(BufWriter::new(std::io::stdout()))),
        Some(p) => Ok(Box::new(BufWriter::new(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(p)?,
        ))),
    }
}
/// Accept common NCBI-style -query/-word_size in addition to GNU --query/--word-size.
pub fn arguments() -> Vec<OsString> {
    std::env::args_os()
        .enumerate()
        .map(|(i, a)| {
            if i == 0 {
                return a;
            }
            if let Some(s) = a.to_str()
                && s.starts_with('-')
                && !s.starts_with("--")
                && s.len() > 2
                && s.as_bytes()[1].is_ascii_alphabetic()
            {
                return format!("--{}", s[1..].replace('_', "-")).into();
            }
            a
        })
        .collect()
}
pub fn run(f: impl FnOnce() -> Result<()>) {
    if let Err(Error(message)) = f() {
        eprintln!("error: {message}");
        std::process::exit(2);
    }
}

#[derive(clap::Args, Clone, Copy, Debug)]
pub struct GpuTuning {
    /// Maximum alignments per GPU dispatch (1..1024); does not change DP semantics.
    #[arg(long, default_value_t = 256)]
    pub gpu_batch_size: usize,
    /// Shared GPU buffer slots / maximum uncollected dispatches (1..8).
    #[arg(long, default_value_t = 2)]
    pub gpu_in_flight: usize,
}
impl Default for GpuTuning {
    fn default() -> Self {
        Self {
            gpu_batch_size: 256,
            gpu_in_flight: 2,
        }
    }
}

pub fn alignment_backend(mode: &str) -> Result<Box<dyn blast_core::align::GappedAligner>> {
    alignment_backend_with_tuning(mode, GpuTuning::default())
}

pub fn alignment_backend_with_tuning(
    mode: &str,
    tuning: GpuTuning,
) -> Result<Box<dyn blast_core::align::GappedAligner>> {
    if !(1..=1024).contains(&tuning.gpu_batch_size) || !(1..=8).contains(&tuning.gpu_in_flight) {
        return Err(Error(
            "GPU batch size must be 1..1024 and in-flight slots 1..8".into(),
        ));
    }
    if mode == "scalar" {
        return Ok(Box::new(blast_cpu::ScalarAligner));
    }
    if mode == "simd" {
        return Ok(Box::new(blast_cpu::SimdAligner));
    }
    if mode == "cpu" {
        return Ok(Box::new(blast_cpu::CpuAligner));
    }
    #[cfg(feature = "gpu")]
    {
        match blast_gpu::GpuAligner::with_options(blast_gpu::GpuOptions {
            batch_size: tuning.gpu_batch_size,
            in_flight: tuning.gpu_in_flight,
        }) {
            Ok(gpu) => {
                eprintln!("GPU adapter: {}", gpu.adapter_name);
                if mode == "gpu" {
                    return Ok(Box::new(gpu));
                }
                Ok(Box::new(AdaptiveAligner { gpu }))
            }
            Err(e) if mode == "auto" => {
                eprintln!("{e}; auto uses CPU");
                Ok(Box::new(blast_cpu::CpuAligner))
            }
            Err(e) => Err(e),
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        if mode == "gpu" {
            return Err(Error(
                "GPU support was not compiled; rebuild with --features gpu".into(),
            ));
        }
        Ok(Box::new(blast_cpu::CpuAligner))
    }
}

#[cfg(feature = "gpu")]
struct AdaptiveAligner {
    gpu: blast_gpu::GpuAligner,
}
#[cfg(feature = "gpu")]
impl blast_core::align::GappedAligner for AdaptiveAligner {
    fn name(&self) -> &'static str {
        "auto"
    }
    fn batch_profile(&self) -> Option<blast_core::align::BatchProfile> {
        self.gpu.batch_profile()
    }
    fn score_batch(
        &self,
        requests: &[blast_core::align::AlignRequest<'_>],
        options: blast_core::align::AlignOptions,
    ) -> Result<Vec<blast_core::align::AlignmentScore>> {
        // Conservative initial policy, not a hardware-calibrated performance claim.
        if requests.len() >= 64 {
            self.gpu.score_batch(requests, options)
        } else {
            blast_cpu::CpuAligner.score_batch(requests, options)
        }
    }
    fn traceback_batch(
        &self,
        requests: &[blast_core::align::AlignRequest<'_>],
        options: blast_core::align::AlignOptions,
        scratch: &mut blast_core::align::WorkerScratch,
    ) -> Result<Vec<blast_core::align::Alignment>> {
        // Auto still scores on CPU/GPU by policy; selected tracebacks always run on CPU.
        blast_cpu::CpuAligner.traceback_batch(requests, options, scratch)
    }
}
