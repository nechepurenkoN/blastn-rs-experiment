//! Reproducible whole-process comparison with an installed native NCBI BLAST+.
//! Build both databases once, warm up, rotate the order, retain outputs and raw timings.
use blast_core::{Error, Result};
use clap::Parser;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    ncbi_bin: PathBuf,
    #[arg(long, default_value = "target/release")]
    rust_bin: PathBuf,
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
    #[arg(long, default_value_t = 500)]
    max_target_seqs: usize,
    #[arg(long, default_value = "both", value_parser = ["yes", "no", "both"])]
    dust: String,
    #[arg(long, default_value = "native", value_parser = ["native", "ncbi"])]
    compatibility: String,
    /// Rust backends measured alongside NCBI in the same rotating series.
    #[arg(long, default_value = "cpu", value_delimiter = ',',
          value_parser = ["cpu", "scalar", "simd", "gpu"])]
    backends: Vec<String>,
    #[command(flatten)]
    gpu: blast_cli::GpuTuning,
    /// Fail immediately unless complete TSV bytes (including order/numbers) agree.
    #[arg(long)]
    require_identical: bool,
}

struct Case {
    name: String,
    engine: String,
    program: PathBuf,
    args: Vec<String>,
    dust: bool,
    threads: usize,
}

fn timed(program: &Path, args: &[String], prefix: &Path) -> Result<Value> {
    let mut command = Command::new("/usr/bin/time");
    if cfg!(target_os = "macos") {
        command.arg("-lp");
    } else {
        command.arg("-v");
    }
    command
        .arg(program)
        .args(args)
        .env("BLAST_USAGE_REPORT", "false");
    command.stdout(fs::File::create(prefix.with_extension("stdout"))?);
    command.stderr(fs::File::create(prefix.with_extension("stderr"))?);
    let start = Instant::now();
    let status = command.status()?;
    let elapsed = start.elapsed().as_secs_f64();
    let stderr = fs::read_to_string(prefix.with_extension("stderr"))?;
    if !status.success() {
        return Err(Error(format!("{} failed: {stderr}", program.display())));
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
    Ok(json!({"wall_seconds":elapsed,"peak_rss_bytes":rss,"program":program,"args":args}))
}

fn version(program: &Path, arg: &str) -> Result<String> {
    let out = Command::new(program)
        .arg(arg)
        .env("BLAST_USAGE_REPORT", "false")
        .output()?;
    if !out.status.success() {
        return Err(Error(format!(
            "cannot read version of {}",
            program.display()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

#[derive(Clone)]
struct Hit {
    query: String,
    subject: String,
    q0: usize,
    q1: usize,
    s0: usize,
    s1: usize,
    reverse: bool,
    bits: f64,
    fields: Vec<String>,
}
impl Hit {
    fn key(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.query, self.subject, self.q0, self.q1, self.s0, self.s1, self.reverse
        )
    }
    fn pair(&self) -> (String, String) {
        (self.query.clone(), self.subject.clone())
    }
}
fn hits(path: &Path) -> Result<Vec<Hit>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let fields: Vec<String> = l.split('\t').map(str::to_owned).collect();
            if fields.len() != 12 {
                return Err(Error("expected 12-column BLAST TSV".into()));
            }
            let pos = |i: usize| fields[i].parse::<usize>().map_err(|e| Error(e.to_string()));
            let (qa, qb, sa, sb) = (pos(6)?, pos(7)?, pos(8)?, pos(9)?);
            Ok(Hit {
                query: fields[0].clone(),
                subject: fields[1].clone(),
                q0: qa.min(qb),
                q1: qa.max(qb),
                s0: sa.min(sb),
                s1: sa.max(sb),
                reverse: (qa > qb) != (sa > sb),
                bits: fields[11]
                    .parse::<f64>()
                    .map_err(|e| Error(e.to_string()))?,
                fields,
            })
        })
        .collect()
}
fn compare(reference: &[Hit], rust: &[Hit]) -> Value {
    let refkeys: BTreeSet<_> = reference.iter().map(Hit::key).collect();
    let rustkeys: BTreeSet<_> = rust.iter().map(Hit::key).collect();
    let refpairs: BTreeSet<_> = reference.iter().map(Hit::pair).collect();
    let rustpairs: BTreeSet<_> = rust.iter().map(Hit::pair).collect();
    let mut rust_by_key = BTreeMap::new();
    let mut rust_by_pair = BTreeMap::<(String, String), Vec<&Hit>>::new();
    for h in rust {
        rust_by_key.entry(h.key()).or_insert(h);
        rust_by_pair.entry(h.pair()).or_default().push(h);
    }
    let top = |rows: &[Hit]| {
        let mut best = BTreeMap::<String, (f64, BTreeSet<String>)>::new();
        for h in rows {
            let entry = best
                .entry(h.query.clone())
                .or_insert((f64::NEG_INFINITY, BTreeSet::new()));
            if h.bits > entry.0 {
                entry.0 = h.bits;
                entry.1.clear();
            }
            if h.bits == entry.0 {
                entry.1.insert(h.subject.clone());
            }
        }
        best
    };
    let a = top(reference);
    let b = top(rust);
    let top_agree = a
        .iter()
        .filter(|(q, (_, ids))| b.get(*q).is_some_and(|(_, other)| !ids.is_disjoint(other)))
        .count();
    let overlaps = |h: &Hit, r: &Hit| {
        let iou = |a: usize, b: usize, c: usize, d: usize| {
            let overlap = (b.min(d) + 1).saturating_sub(a.max(c));
            overlap as f64 / (b.max(d) - a.min(c) + 1) as f64
        };
        h.query == r.query
            && h.subject == r.subject
            && h.reverse == r.reverse
            && iou(h.q0, h.q1, r.q0, r.q1) >= 0.9
            && iou(h.s0, h.s1, r.s0, r.s1) >= 0.9
    };
    let matched = reference
        .iter()
        .filter(|h| {
            rust_by_pair
                .get(&h.pair())
                .is_some_and(|rows| rows.iter().any(|r| overlaps(h, r)))
        })
        .count();
    let shared: Vec<_> = reference
        .iter()
        .filter_map(|h| rust_by_key.get(&h.key()).map(|&r| (h, r)))
        .collect();
    let alignment_fields_agree = shared
        .iter()
        .filter(|(h, r)| h.fields[3..6] == r.fields[3..6])
        .count();
    json!({"ncbi_hsps":reference.len(),"rust_hsps":rust.len(),"exact_coordinate_hsps":refkeys.intersection(&rustkeys).count(),"ncbi_hsps_with_90pct_interval_iou":matched,"shared_coordinates_with_equal_length_mismatch_gapopen":alignment_fields_agree,"ncbi_subject_pairs":refpairs.len(),"rust_subject_pairs":rustpairs.len(),"shared_subject_pairs":refpairs.intersection(&rustpairs).count(),"queries_with_ncbi_hits":a.len(),"top_subject_agreement":top_agree,"ncbi_top_subjects":a,"rust_top_subjects":b})
}

fn main() {
    blast_cli::run(run);
}
fn run() -> Result<()> {
    let args = Args::parse();
    if args.repeats < 3 || args.repeats > 100 || args.warmups > 10 {
        return Err(Error("use 3..100 repeats and 0..10 warmups".into()));
    }
    if args.backends.iter().collect::<BTreeSet<_>>().len() != args.backends.len() {
        return Err(Error("backend names must be unique".into()));
    }
    if !(1..=1024).contains(&args.gpu.gpu_batch_size) || !(1..=8).contains(&args.gpu.gpu_in_flight)
    {
        return Err(Error(
            "GPU batch size 1..1024 and in-flight slots 1..8 required".into(),
        ));
    }
    let ncbi = args.ncbi_bin.canonicalize()?;
    let rust = args.rust_bin.canonicalize()?;
    let query = args.query.canonicalize()?;
    let subject = args.subject.canonicalize()?;
    fs::create_dir(&args.output_dir)?;
    let dir = args.output_dir.canonicalize()?;
    let ncbi_db = dir.join("ncbi-db");
    let rust_db = dir.join("rust-db.rdb");
    let n = ncbi.join("blastn");
    let r = rust.join("blastn-rs");
    let build_ncbi = timed(
        &ncbi.join("makeblastdb"),
        &[
            "-in".into(),
            subject.display().to_string(),
            "-dbtype".into(),
            "nucl".into(),
            "-parse_seqids".into(),
            "-out".into(),
            ncbi_db.display().to_string(),
        ],
        &dir.join("build-ncbi"),
    )?;
    let build_rust = timed(
        &rust.join("makedb-rs"),
        &[
            "--in".into(),
            subject.display().to_string(),
            "--out".into(),
            rust_db.display().to_string(),
        ],
        &dir.join("build-rust"),
    )?;
    let mut cases = Vec::new();
    for dust in [false, true] {
        if (args.dust == "no" && dust) || (args.dust == "yes" && !dust) {
            continue;
        }
        for threads in [1usize, 4] {
            for backend in std::iter::once("ncbi").chain(args.backends.iter().map(String::as_str)) {
                let original = backend == "ncbi";
                let engine = if original {
                    "ncbi".to_string()
                } else {
                    format!("rust-{backend}")
                };
                let mut options: Vec<String> = if original {
                    [
                        "-task",
                        "blastn",
                        "-word_size",
                        "11",
                        "-reward",
                        "2",
                        "-penalty",
                        "-3",
                        "-gapopen",
                        "5",
                        "-gapextend",
                        "2",
                        "-strand",
                        "both",
                        "-evalue",
                        "10",
                        "-outfmt",
                        "6",
                        "-window_size",
                        "0",
                        "-xdrop_ungap",
                        "20",
                        "-xdrop_gap",
                        "50",
                        "-xdrop_gap_final",
                        "50",
                        "-soft_masking",
                        "true",
                        "-no_greedy",
                        "-mt_mode",
                        "0",
                    ]
                    .map(str::to_owned)
                    .into()
                } else {
                    [
                        "--task",
                        "blastn",
                        "--word-size",
                        "11",
                        "--reward",
                        "2",
                        "--penalty",
                        "-3",
                        "--gapopen",
                        "5",
                        "--gapextend",
                        "2",
                        "--strand",
                        "both",
                        "--evalue",
                        "10",
                        "--outfmt",
                        "6",
                        "--window-size",
                        "0",
                        "--xdrop-ungap",
                        "20",
                        "--xdrop-gap",
                        "50",
                        "--backend",
                        backend,
                        "--band",
                        if args.compatibility == "ncbi" {
                            "0"
                        } else {
                            "64"
                        },
                    ]
                    .map(str::to_owned)
                    .into()
                };
                if !original && args.compatibility != "native" {
                    options.extend(["--compatibility".into(), args.compatibility.clone()]);
                }
                if backend == "gpu" {
                    options.extend([
                        "--gpu-batch-size".into(),
                        args.gpu.gpu_batch_size.to_string(),
                        "--gpu-in-flight".into(),
                        args.gpu.gpu_in_flight.to_string(),
                    ]);
                }
                let (qflag, dflag, tflag, mflag) = if original {
                    ("-query", "-db", "-num_threads", "-dust")
                } else {
                    ("--query", "--db", "--num-threads", "--dust")
                };
                options.extend([
                    if original {
                        "-max_target_seqs".into()
                    } else {
                        "--max-target-seqs".into()
                    },
                    args.max_target_seqs.to_string(),
                    qflag.into(),
                    query.display().to_string(),
                    dflag.into(),
                    if original {
                        ncbi_db.display().to_string()
                    } else {
                        rust_db.display().to_string()
                    },
                    tflag.into(),
                    threads.to_string(),
                    mflag.into(),
                    if dust { "yes".into() } else { "no".into() },
                ]);
                cases.push(Case {
                    name: format!(
                        "{}-{}-t{threads}",
                        engine,
                        if dust { "dust" } else { "no-dust" }
                    ),
                    engine,
                    program: if original { n.clone() } else { r.clone() },
                    args: options,
                    dust,
                    threads,
                });
            }
        }
    }
    let mut samples: Vec<Vec<Value>> = vec![Vec::new(); cases.len()];
    let mut expected: Vec<Option<Vec<u8>>> = vec![None; cases.len()];
    for round in 0..args.warmups + args.repeats {
        for position in 0..cases.len() {
            let index = (position + round) % cases.len();
            let case = &cases[index];
            let stem = dir.join(format!("{}-r{round}", case.name));
            let output = stem.with_extension("tsv");
            let mut command = case.args.clone();
            command.extend([
                if case.name.starts_with("ncbi") {
                    "-out".into()
                } else {
                    "--out".into()
                },
                output.display().to_string(),
            ]);
            let sample = timed(&case.program, &command, &stem)?;
            eprintln!(
                "{} round {round}: {:.4}s",
                case.name,
                sample["wall_seconds"].as_f64().unwrap()
            );
            let result = fs::read(&output)?;
            if let Some(old) = &expected[index] {
                if old != &result {
                    return Err(Error(format!("nondeterministic output from {}", case.name)));
                }
            } else {
                expected[index] = Some(result);
            }
            if round >= args.warmups {
                samples[index].push(sample);
            }
        }
        if args.require_identical {
            for (index, case) in cases.iter().enumerate() {
                let reference = cases
                    .iter()
                    .position(|c| {
                        c.engine == "ncbi" && c.dust == case.dust && c.threads == case.threads
                    })
                    .unwrap();
                if expected[reference] != expected[index] {
                    return Err(Error(format!(
                        "NCBI/{} TSV differs in round {round}; inspect saved TSV artifacts",
                        case.engine
                    )));
                }
            }
        }
    }
    let mut summaries = Vec::new();
    for (case, samples) in cases.iter().zip(samples) {
        let mut times: Vec<f64> = samples
            .iter()
            .map(|v| v["wall_seconds"].as_f64().unwrap())
            .collect();
        let mid = median(&mut times);
        let mut memory: Vec<f64> = samples
            .iter()
            .filter_map(|v| {
                v["peak_rss_bytes"]
                    .as_u64()
                    .map(|n| n as f64 / 1024.0 / 1024.0)
            })
            .collect();
        let rss = if memory.is_empty() {
            None
        } else {
            Some(median(&mut memory))
        };
        summaries.push(json!({"name":case.name,"engine":case.engine,"dust":case.dust,"threads":case.threads,"median_seconds":mid,"min_seconds":times[0],"max_seconds":times[times.len()-1],"median_peak_rss_mib":rss,"samples":samples}));
        println!(
            "{}\t{mid:.6}\t{:.6}\t{:.6}\t{rss:?}",
            case.name,
            times[0],
            times[times.len() - 1]
        );
    }
    let mut comparisons = BTreeMap::new();
    for dust in ["no-dust", "dust"] {
        if (args.dust == "no" && dust == "dust") || (args.dust == "yes" && dust == "no-dust") {
            continue;
        }
        let a = hits(&dir.join(format!("ncbi-{dust}-t1-r0.tsv")))?;
        for backend in &args.backends {
            let b = hits(&dir.join(format!("rust-{backend}-{dust}-t1-r0.tsv")))?;
            let mut comparison = compare(&a, &b);
            comparison["tsv_byte_identical"] = json!(
                fs::read(dir.join(format!("ncbi-{dust}-t1-r0.tsv")))?
                    == fs::read(dir.join(format!("rust-{backend}-{dust}-t1-r0.tsv")))?
            );
            comparisons.insert(format!("{backend}-{dust}"), comparison);
        }
        for engine in std::iter::once("ncbi".to_string())
            .chain(args.backends.iter().map(|b| format!("rust-{b}")))
        {
            if fs::read(dir.join(format!("{engine}-{dust}-t1-r0.tsv")))?
                != fs::read(dir.join(format!("{engine}-{dust}-t4-r0.tsv")))?
            {
                return Err(Error(format!(
                    "thread count changes output for {engine}/{dust}"
                )));
            }
        }
    }
    let report = json!({"ncbi_version":version(&n,"-version")?,"rust_version":version(&r,"--version")?,"compatibility":args.compatibility,"backends":args.backends,"gpu_batch_size":args.gpu.gpu_batch_size,"gpu_in_flight":args.gpu.gpu_in_flight,"require_identical":args.require_identical,"query":query,"subject":subject,"repeats":args.repeats,"warmups":args.warmups,"max_target_seqs":args.max_target_seqs,"build_ncbi":build_ncbi,"build_rust":build_rust,"cases":summaries,"comparisons":comparisons,"note":"Matched explicit controls: task blastn, non-greedy DP, X-drop 20/50/50, soft masking, one-hit. NCBI profile uses unbanded DP and NCBI cutoffs/order/numbers; native retains band 64 and legacy behavior. Neither mode implies general algorithm parity. Warm OS caches. Timings include startup, GPU initialization when selected, input, search, formatting and output; exclude database construction. GPU thread counts refer to CPU host workers; RSS is process resident memory, not total GPU memory."});
    fs::write(
        dir.join("summary.json"),
        serde_json::to_string_pretty(&report).map_err(|e| Error(e.to_string()))?,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report["comparisons"]).map_err(|e| Error(e.to_string()))?
    );
    Ok(())
}
