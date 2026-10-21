//! Compare two Rust binaries on identical data; fail on any byte-level output difference.
use blast_core::{Error, Result};
use blast_db::{DatabaseBuilder, FastaReader};
use clap::Parser;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long, default_value = "target/release/blastn-rs")]
    candidate: PathBuf,
    #[arg(long)]
    query: PathBuf,
    #[arg(long)]
    subject: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 7)]
    repeats: usize,
    #[arg(long, default_value_t = 1)]
    warmups: usize,
    #[arg(long, default_value = "cpu", value_parser = ["cpu", "gpu"])]
    backend: String,
    /// Both binaries must support the selected compatibility profile.
    #[arg(long, default_value = "native", value_parser = ["native", "ncbi"])]
    compatibility: String,
    #[arg(long, value_delimiter = ',', default_value = "1,4")]
    threads: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "1,32")]
    query_batches: Vec<usize>,
    #[arg(long)]
    full_only: bool,
    /// Add a current CPU case per thread count when comparing GPU binaries.
    #[arg(long)]
    cpu_reference: bool,
    /// Pass the same profiling/batch/tuning flags to a baseline supporting current CLI options.
    #[arg(long)]
    matched_baseline_options: bool,
    #[command(flatten)]
    gpu: blast_cli::GpuTuning,
}
struct Case {
    name: String,
    program: PathBuf,
    args: Vec<String>,
    limit: usize,
    batch: usize,
    threads: usize,
    backend: String,
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let i = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[i - 1] + values[i]) / 2.0
    } else {
        values[i]
    }
}
fn timed(case: &Case, prefix: &Path, format: &str) -> Result<(Value, Vec<u8>, String)> {
    let args: Vec<_> = case
        .args
        .iter()
        .cloned()
        .chain(["--outfmt".into(), format.into()])
        .collect();
    let mut command = Command::new("/usr/bin/time");
    if cfg!(target_os = "macos") {
        command.arg("-lp");
    } else {
        command.arg("-v");
    }
    command
        .arg(&case.program)
        .args(&args)
        .stdout(fs::File::create(prefix.with_extension(format))?)
        .stderr(fs::File::create(prefix.with_extension("stderr"))?);
    let start = Instant::now();
    let status = command.status()?;
    let wall = start.elapsed().as_secs_f64();
    let stderr = fs::read_to_string(prefix.with_extension("stderr"))?;
    if !status.success() {
        return Err(Error(format!("{} failed: {stderr}", case.name)));
    }
    let rss = stderr.lines().find_map(|line| {
        if line.contains("maximum resident set size") {
            line.split_whitespace().next()?.parse::<u64>().ok()
        } else if line.contains("Maximum resident set size (kbytes)") {
            Some(line.split_whitespace().last()?.parse::<u64>().ok()? * 1024)
        } else {
            None
        }
    });
    let work = stderr
        .lines()
        .find(|line| line.contains(" queries;"))
        .ok_or_else(|| Error("missing search diagnostics".into()))?
        .split(';')
        .take(4)
        .collect::<Vec<_>>()
        .join(";");
    let output = fs::read(prefix.with_extension(format))?;
    Ok((
        json!({"wall_seconds":wall,"peak_rss_bytes":rss,"program":case.program,"args":args}),
        output,
        work,
    ))
}
fn main() {
    blast_cli::run(run);
}
fn run() -> Result<()> {
    let args = Args::parse();
    if !(3..=100).contains(&args.repeats) || args.warmups > 10 {
        return Err(Error("use 3..100 repeats and 0..10 warmups".into()));
    }
    if args.threads.is_empty()
        || args.threads.contains(&0)
        || args.query_batches.is_empty()
        || args.query_batches.contains(&0)
    {
        return Err(Error(
            "positive thread counts and candidate query batch sizes required".into(),
        ));
    }
    if args.cpu_reference && args.backend != "gpu" {
        return Err(Error("--cpu-reference requires --backend gpu".into()));
    }
    if args.matched_baseline_options && args.query_batches.len() != 1 {
        return Err(Error(
            "--matched-baseline-options requires exactly one query batch size".into(),
        ));
    }
    let baseline = args.baseline.canonicalize()?;
    let candidate = args.candidate.canonicalize()?;
    let query = args.query.canonicalize()?;
    let subject = args.subject.canonicalize()?;
    fs::create_dir(&args.output_dir)?;
    let dir = args.output_dir.canonicalize()?;
    let db = dir.join("subject.rdb");
    let (subjects, bases) =
        DatabaseBuilder::build(FastaReader::new(blast_cli::input(&subject)?), &db)?;
    let mut cases = Vec::new();
    let limits = if args.full_only {
        vec![subjects.max(500) + 1]
    } else {
        vec![500, subjects.max(500) + 1]
    };
    for limit in limits {
        for &threads in &args.threads {
            let configurations = std::iter::once((0, args.backend.as_str()))
                .chain(
                    args.query_batches
                        .iter()
                        .map(|&batch| (batch, args.backend.as_str())),
                )
                .chain(args.cpu_reference.then_some((32, "cpu")));
            for (batch, backend) in configurations {
                let mut options = vec![
                    "--query".into(),
                    query.display().to_string(),
                    "--db".into(),
                    db.display().to_string(),
                    "--backend".into(),
                    backend.into(),
                    "--dust".into(),
                    "no".into(),
                    "--max-target-seqs".into(),
                    limit.to_string(),
                    "--num-threads".into(),
                    threads.to_string(),
                ];
                // Preserve support for old native-only baseline executables.
                if args.compatibility != "native" {
                    options.extend(["--compatibility".into(), args.compatibility.clone()]);
                }
                let effective_batch = if batch == 0 && args.matched_baseline_options {
                    args.query_batches[0]
                } else {
                    batch
                };
                if effective_batch != 0 {
                    options.extend([
                        "--query-batch-size".into(),
                        effective_batch.to_string(),
                        "--profile".into(),
                    ]);
                    if backend == "gpu" {
                        options.extend([
                            "--gpu-batch-size".into(),
                            args.gpu.gpu_batch_size.to_string(),
                            "--gpu-in-flight".into(),
                            args.gpu.gpu_in_flight.to_string(),
                        ]);
                    }
                }
                cases.push(Case {
                    name: format!(
                        "{}-limit{limit}-t{threads}-batch{batch}",
                        if batch == 0 {
                            "before"
                        } else if args.cpu_reference && backend == "cpu" {
                            "cpu"
                        } else {
                            "after"
                        }
                    ),
                    program: if batch == 0 {
                        baseline.clone()
                    } else {
                        candidate.clone()
                    },
                    args: options,
                    limit,
                    batch: effective_batch,
                    threads,
                    backend: backend.into(),
                });
            }
        }
    }
    // JSON checks exact raw scores, float values, CIGAR, strand and band diagnostics as well as TSV fields.
    let mut json_references = BTreeMap::<usize, (Vec<u8>, String)>::new();
    for case in &cases {
        let (_, bytes, work) = timed(case, &dir.join(format!("{}-validation", case.name)), "json")?;
        match json_references.get(&case.limit) {
            Some((reference, expected_work)) if *reference != bytes || *expected_work != work => {
                return Err(Error(format!(
                    "JSON or search-work mismatch: {}",
                    case.name
                )));
            }
            None => {
                json_references.insert(case.limit, (bytes, work));
            }
            _ => {}
        }
    }
    eprintln!(
        "JSON and search-work validation passed for all {} configurations",
        cases.len()
    );
    let mut references = BTreeMap::<usize, Vec<u8>>::new();
    let mut samples = BTreeMap::<String, Vec<Value>>::new();
    for round in 0..args.warmups + args.repeats {
        for offset in 0..cases.len() {
            let case = &cases[(offset + round) % cases.len()];
            let (sample, bytes, work) =
                timed(case, &dir.join(format!("{}-r{round}", case.name)), "6")?;
            if let Some(expected) = references.get(&case.limit) {
                if *expected != bytes {
                    return Err(Error(format!("TSV mismatch: {} round {round}", case.name)));
                }
            } else {
                references.insert(case.limit, bytes);
            }
            if json_references[&case.limit].1 != work {
                return Err(Error(format!(
                    "search-work mismatch: {} round {round}",
                    case.name
                )));
            }
            if round >= args.warmups {
                samples.entry(case.name.clone()).or_default().push(sample);
            }
        }
        eprintln!(
            "round {}/{} complete; all outputs identical",
            round + 1,
            args.warmups + args.repeats
        );
    }
    let mut summaries = BTreeMap::new();
    for case in &cases {
        let samples = &samples[&case.name];
        let mut walls: Vec<_> = samples
            .iter()
            .map(|s| s["wall_seconds"].as_f64().unwrap())
            .collect();
        let wall = median(&mut walls);
        let mut rss: Vec<_> = samples
            .iter()
            .filter_map(|s| s["peak_rss_bytes"].as_f64())
            .collect();
        let rss_mib = if rss.is_empty() {
            None
        } else {
            Some(median(&mut rss) / 1048576.0)
        };
        eprintln!("{}: {wall:.6}s; RSS {rss_mib:?} MiB", case.name);
        summaries.insert(case.name.clone(), json!({
            "threads":case.threads,"query_batch_size":case.batch,"max_target_seqs":case.limit,"backend":case.backend,
            "wall_median_seconds":wall,"wall_min_seconds":walls[0],"wall_max_seconds":walls[walls.len()-1],
            "peak_rss_median_mib":rss_mib,"samples":samples,
        }));
    }
    let report = json!({
        "baseline":baseline,"candidate":candidate,"query":query,"subject":subject,
        "compatibility":args.compatibility,
        "subjects":subjects,"subject_bases":bases,"warmups":args.warmups,"repeats":args.repeats,
        "method":"sequential whole-process wall time; rotated order; warm OS cache; excludes DB build",
        "tsv_byte_identical":true,"json_byte_identical":true,"search_work_identical":true,
        "cases":summaries,
    });
    fs::write(
        dir.join("summary.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| Error(e.to_string()))?,
    )?;
    Ok(())
}
