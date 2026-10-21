use blast_core::{
    Error,
    align::{AlignOptions, ScoringScheme},
    search::{HitLimits, PreparedQuery, SearchDiagnostics, SearchOptions, StrandSelection},
    stats::{KarlinStats, UngappedStats},
};
use blast_cpu::SearchEngine;
use blast_db::{BlastDatabase, FastaReader, MemoryStore, SequenceStore};
use blast_format::{Formatter, OutputFormat, ResultFormatter};
use clap::Parser;
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(
    name = "blastn-rs",
    version,
    about = "Rust nucleotide seed-and-extend search; experimental BLASTN analogue",
    after_help = "Supports task blastn and RBLAST01 databases. Not a bit-for-bit NCBI BLAST+ replacement. X-drop values are in bits. Gap cost = open + length*extend."
)]
struct Args {
    #[arg(long, default_value="native", value_parser=["native", "ncbi"],
        help="Experimental NCBI BLASTN cutoffs/order/format profile; not full BLAST+ compatibility")]
    compatibility: String,
    #[arg(long)]
    query: PathBuf,
    #[arg(long, required_unless_present = "subject", conflicts_with = "subject")]
    db: Option<PathBuf>,
    #[arg(long, required_unless_present = "db")]
    subject: Option<PathBuf>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long,default_value="blastn",value_parser=["blastn"])]
    task: String,
    #[arg(long, default_value_t = 11)]
    word_size: usize,
    #[arg(long,default_value="both",value_parser=["both","plus","minus"])]
    strand: String,
    #[arg(long,default_value="yes",value_parser=["yes","no"])]
    dust: String,
    #[arg(long)]
    lcase_masking: bool,
    #[arg(long, default_value_t = 2)]
    reward: i32,
    #[arg(long,default_value_t=-3,allow_hyphen_values=true)]
    penalty: i32,
    #[arg(long, default_value_t = 5)]
    gapopen: i32,
    #[arg(long, default_value_t = 2)]
    gapextend: i32,
    #[arg(long, default_value_t = 20.0)]
    xdrop_ungap: f64,
    #[arg(long, default_value_t = 50.0)]
    xdrop_gap: f64,
    #[arg(long)]
    min_ungapped_score: Option<i32>,
    #[arg(long, default_value_t = 0)]
    min_score: i32,
    #[arg(long, default_value_t = 10.0)]
    evalue: f64,
    #[arg(
        long,
        help = "DP half-band width; 0 = full subject flank; default 64 (native), 0 (ncbi)"
    )]
    band: Option<usize>,
    #[arg(long, default_value_t = 0)]
    window_size: usize,
    #[arg(
        long,
        default_value_t = 0,
        help = "Suppress words occurring more often in query; 0 disables"
    )]
    max_seed_occurrences: usize,
    #[arg(long, default_value_t = 100_000)]
    max_candidates: usize,
    #[arg(long, default_value_t = 20_000_000)]
    max_cells: usize,
    #[arg(long, default_value_t = 500)]
    max_target_seqs: usize,
    #[arg(long, default_value_t = 0)]
    max_hsps: usize,
    #[arg(long, default_value_t = 1)]
    num_threads: usize,
    #[arg(
        long,
        default_value_t = 32,
        help = "Maximum queries per shared subject scan"
    )]
    query_batch_size: usize,
    #[arg(
        long,
        default_value_t = 65_536,
        help = "Soft query-base budget per batch (one record may exceed it)"
    )]
    query_batch_bases: usize,
    #[arg(
        long,
        default_value_t = 128,
        help = "Subject records per cross-subject score batch"
    )]
    subject_batch_size: usize,
    #[arg(
        long,
        default_value_t = 1_048_576,
        help = "Soft subject-base budget per worker block; records are never split"
    )]
    subject_batch_bases: usize,
    #[arg(
        long,
        help = "Print stage wall times and subject decode counts to stderr"
    )]
    profile: bool,
    #[arg(long,default_value="auto",value_parser=["auto","cpu","scalar","simd","gpu"])]
    backend: String,
    #[command(flatten)]
    gpu: blast_cli::GpuTuning,
    #[arg(long,default_value="6",value_parser=["6","7","json"])]
    outfmt: String,
}
fn main() {
    blast_cli::run(|| {
        let args = Args::parse_from(blast_cli::arguments());
        let started = Instant::now();
        for input in [Some(&args.query), args.db.as_ref(), args.subject.as_ref()]
            .into_iter()
            .flatten()
        {
            if args.out.as_ref() == Some(input) {
                return Err(Error("output path must differ from all input paths".into()));
            }
        }
        if args.query == std::path::Path::new("-")
            && args
                .subject
                .as_ref()
                .is_some_and(|p| p == std::path::Path::new("-"))
        {
            return Err(Error("query and subject cannot both read stdin".into()));
        }
        if args.max_target_seqs == 0
            || args.query_batch_size == 0
            || args.query_batch_bases == 0
            || !args.xdrop_ungap.is_finite()
            || !args.xdrop_gap.is_finite()
            || !(0.0..=10_000.0).contains(&args.xdrop_ungap)
            || !(0.0..=10_000.0).contains(&args.xdrop_gap)
        {
            return Err(Error("invalid hit limit or X-drop threshold".into()));
        }
        let scoring = ScoringScheme {
            reward: args.reward,
            penalty: args.penalty,
            gap_open: args.gapopen,
            gap_extend: args.gapextend,
        };
        let stats = KarlinStats::for_scoring(scoring)?;
        let ncbi = args.compatibility == "ncbi";
        if ncbi && (args.min_ungapped_score.is_some() || args.band.is_some_and(|b| b != 0)) {
            return Err(Error("ncbi profile derives its ungapped cutoff and requires --band 0; omit --min-ungapped-score".into()));
        }
        let band = args.band.unwrap_or(if ncbi { 0 } else { 64 });
        let options = SearchOptions {
            ncbi_compatibility: ncbi,
            word_size: args.word_size,
            strand: match args.strand.as_str() {
                "plus" => StrandSelection::Plus,
                "minus" => StrandSelection::Minus,
                _ => StrandSelection::Both,
            },
            dust: args.dust == "yes",
            lowercase_masking: args.lcase_masking,
            two_hit_window: args.window_size,
            max_seed_occurrences: args.max_seed_occurrences,
            ungapped_xdrop: if ncbi {
                UngappedStats::for_scoring(scoring)?.raw_drop(args.xdrop_ungap)
            } else {
                stats.raw_drop(args.xdrop_ungap)
            },
            min_ungapped_score: args.min_ungapped_score.unwrap_or(16),
            min_score: args.min_score,
            evalue: args.evalue,
            max_candidates: args.max_candidates,
            align: AlignOptions {
                band: if band == 0 { None } else { Some(band) },
                xdrop: stats.raw_drop(args.xdrop_gap).max(if ncbi {
                    scoring.gap_open + scoring.gap_extend
                } else {
                    0
                }),
                max_cells: args.max_cells,
                scoring,
            },
        };
        options.validate()?;
        let db: Box<dyn SequenceStore> = if let Some(path) = args.db {
            Box::new(BlastDatabase::open(path)?)
        } else {
            Box::new(MemoryStore::new(
                FastaReader::new(blast_cli::input(args.subject.as_ref().unwrap())?)
                    .collect::<blast_core::Result<Vec<_>>>()?,
            ))
        };
        if db.is_empty() {
            return Err(Error("empty subject database".into()));
        }
        let engine = SearchEngine::new(args.num_threads, args.backend != "scalar")?
            .with_subject_batching(args.subject_batch_size, args.subject_batch_bases)?;
        let backend = blast_cli::alignment_backend_with_tuning(&args.backend, args.gpu)?;
        let input = blast_cli::input(&args.query)?;
        let output = blast_cli::output(args.out.as_deref())?;
        let mut formatter = Formatter::new(
            output,
            match args.outfmt.as_str() {
                "7" => OutputFormat::Commented,
                "json" => OutputFormat::Json,
                _ => OutputFormat::Tabular,
            },
        )?
        .with_ncbi_numbers(ncbi);
        let mut count = 0;
        let mut diagnostics = SearchDiagnostics::default();
        let mut reader = FastaReader::new(input);
        loop {
            let prepare_start = Instant::now();
            let mut queries = Vec::new();
            let mut bases = 0;
            for query in reader.by_ref().take(args.query_batch_size) {
                let query = query?;
                bases += query.sequence.len();
                let prepared =
                    PreparedQuery::new(&query, options.clone(), db.total_bases(), db.len())?;
                let suppressed: usize = prepared
                    .contexts
                    .iter()
                    .map(|c| c.lookup.suppressed_words)
                    .sum();
                if suppressed > 0 {
                    eprintln!(
                        "{}: suppressed {suppressed} repetitive query words",
                        query.id
                    );
                }
                queries.push(prepared);
                if bases >= args.query_batch_bases {
                    break;
                }
            }
            if queries.is_empty() {
                break;
            }
            let prepare_seconds = prepare_start.elapsed().as_secs_f64();
            let (results, timings) = engine.search_batch_profiled(
                &queries,
                db.as_ref(),
                backend.as_ref(),
                HitLimits {
                    max_target_seqs: args.max_target_seqs,
                    max_hsps: args.max_hsps,
                },
            )?;
            if args.profile {
                eprintln!(
                    "profile: queries={} prepare={prepare_seconds:.6}s score={:.6}s select={:.6}s traceback={:.6}s decoded_score={} decoded_traceback={}",
                    queries.len(),
                    timings.scoring_seconds,
                    timings.selection_seconds,
                    timings.traceback_seconds,
                    timings.scoring_subjects_decoded,
                    timings.traceback_subjects_decoded
                );
            }
            for (query, (hits, d)) in queries.iter().zip(results) {
                diagnostics.merge(&d);
                formatter.write_query(&query.id, &hits)?;
                count += 1;
            }
        }
        if count == 0 {
            return Err(Error("empty query input".into()));
        }
        formatter.finish()?;
        if args.profile
            && let Some(p) = backend.batch_profile()
        {
            let distribution = |d: &blast_core::align::BatchDistribution| {
                serde_json::json!({
                    "count": d.count, "total": d.total, "max": d.max,
                    "histogram": d.histogram,
                })
            };
            eprintln!(
                "profile_backend: {}",
                serde_json::json!({
                    "backend": p.backend, "cumulative": true,
                    "bins": ["0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256-511", "512-1023", "1024+"],
                    "calls": distribution(&p.calls), "dispatches": distribution(&p.dispatches),
                    "query_flanks": distribution(&p.query_flanks), "subject_bases": p.subject_bases,
                    "bytes_uploaded": p.bytes_uploaded, "buffer_allocations": p.buffer_allocations,
                    "bind_groups_created": p.bind_groups_created,
                })
            );
        }
        eprintln!(
            "{count} queries; {} seeds; {} ungapped extensions; {} gapped candidates; {} tracebacks; {:.3}s",
            diagnostics.seed_hits,
            diagnostics.ungapped_extensions,
            diagnostics.gapped_candidates,
            diagnostics.tracebacks,
            started.elapsed().as_secs_f64()
        );
        if diagnostics.band_edge_candidates > 0 {
            eprintln!(
                "{} candidates reached the DP band edge; widen --band or use --band 0 to check sensitivity",
                diagnostics.band_edge_candidates
            );
        }
        Ok(())
    });
}
