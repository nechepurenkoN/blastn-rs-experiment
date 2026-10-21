use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "blast-rs-e2e-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn file(&self, name: &str, text: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, text).unwrap();
        path
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn execute(binary: &str, args: &[&str], dir: &Path) -> Output {
    Command::new(binary)
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap()
}
fn success(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
const SEARCH: &str = env!("CARGO_BIN_EXE_blastn-rs");
const MAKEDB: &str = env!("CARGO_BIN_EXE_makedb-rs");
const DBCMD: &str = env!("CARGO_BIN_EXE_dbcmd-rs");

#[test]
fn gap_2_1_evalue_boundary_preserves_significant_hit() {
    let tmp = Temp::new();
    let query = "TAGTGATAAGTGGAATTGATGGGGTTTGTGTAGATATTATGTTTGGATGT";
    tmp.file("q.fa", &format!(">q\n{query}\n"));
    let mut subject = String::new();
    for i in 0..100 {
        subject.push_str(&format!(">s{i}\n"));
        if i == 0 {
            subject.push_str(&query[..20]);
            subject.push_str(&"C".repeat(9980));
        } else {
            subject.push_str(&"C".repeat(10_000));
        }
        subject.push('\n');
    }
    tmp.file("s.fa", &subject);
    for profile in ["native", "ncbi"] {
        for threshold in ["0.00005", "0.000049"] {
            let out = execute(
                SEARCH,
                &[
                    "--query",
                    "q.fa",
                    "--subject",
                    "s.fa",
                    "--backend",
                    "cpu",
                    "--compatibility",
                    profile,
                    "--strand",
                    "plus",
                    "--dust",
                    "no",
                    "--reward",
                    "1",
                    "--penalty",
                    "-3",
                    "--gapopen",
                    "2",
                    "--gapextend",
                    "1",
                    "--evalue",
                    threshold,
                    "--outfmt",
                    "json",
                ],
                &tmp.0,
            );
            success(&out);
            let rows: Vec<serde_json::Value> = serde_json::from_slice(&out.stdout).unwrap();
            if threshold == "0.000049" {
                assert!(rows.is_empty());
            } else {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0]["score"], 20);
                assert_eq!(rows[0]["cigar"], "20=");
                let evalue = rows[0]["evalue"].as_f64().unwrap();
                assert!((evalue - 4.9516871490926656e-5).abs() < 1e-16);
            }
        }
    }
}

#[test]
fn ncbi_profile_matches_independent_blast_217_golden_tsv() {
    let tmp = Temp::new();
    tmp.file("q.fa", include_str!("fixtures/ncbi/query.fa"));
    tmp.file("s.fa", include_str!("fixtures/ncbi/subject.fa"));
    success(&execute(
        MAKEDB,
        &["--in", "s.fa", "--out", "s.rdb"],
        &tmp.0,
    ));
    let cases: &[(&[&str], &str)] = &[
        (&["--dust", "no"], include_str!("fixtures/ncbi/no.tsv")),
        (&["--dust", "yes"], include_str!("fixtures/ncbi/yes.tsv")),
        (
            &["--dust", "no", "--max-target-seqs", "1"],
            include_str!("fixtures/ncbi/one.tsv"),
        ),
        (
            &[
                "--dust",
                "yes",
                "--reward",
                "1",
                "--penalty",
                "-3",
                "--gapopen",
                "2",
            ],
            include_str!("fixtures/ncbi/score13.tsv"),
        ),
        (
            &[
                "--dust",
                "yes",
                "--reward",
                "1",
                "--penalty",
                "-3",
                "--gapopen",
                "2",
                "--gapextend",
                "1",
            ],
            include_str!("fixtures/ncbi/score13-gap21.tsv"),
        ),
        (
            &["--dust", "yes", "--strand", "minus"],
            include_str!("fixtures/ncbi/minus.tsv"),
        ),
    ];
    for (extra, expected) in cases {
        for (backend, threads, batch, source) in [
            ("scalar", "1", "1", "--subject"),
            ("cpu", "4", "32", "--db"),
            ("simd", "4", "1", "--db"),
        ] {
            let mut args = vec![
                "--compatibility",
                "ncbi",
                "--query",
                "q.fa",
                source,
                if source == "--db" { "s.rdb" } else { "s.fa" },
                "--backend",
                backend,
                "--num-threads",
                threads,
                "--query-batch-size",
                batch,
            ];
            args.extend_from_slice(extra);
            let out = execute(SEARCH, &args, &tmp.0);
            success(&out);
            assert_eq!(
                String::from_utf8(out.stdout).unwrap(),
                *expected,
                "{args:?}"
            );
        }
    }
}

#[test]
fn ncbi_profile_rejects_incompatible_controls_and_ambiguous_dust() {
    let tmp = Temp::new();
    tmp.file("q.fa", ">q\nACGTNNNNACGTACGTACGT\n");
    tmp.file("s.fa", ">s\nACGTACGTACGTACGTACGT\n");
    for extra in [
        vec!["--band", "64"],
        vec!["--min-ungapped-score", "16"],
        vec!["--dust", "yes"],
    ] {
        let mut args = vec![
            "--compatibility",
            "ncbi",
            "--query",
            "q.fa",
            "--subject",
            "s.fa",
            "--backend",
            "cpu",
        ];
        args.extend(extra);
        let out = execute(SEARCH, &args, &tmp.0);
        assert!(!out.status.success());
        assert!(
            String::from_utf8_lossy(&out.stderr)
                .to_lowercase()
                .contains("ncbi")
        );
    }
}

#[test]
fn ncbi_profile_raw_scores_and_traceback_match_btop_oracle() {
    let tmp = Temp::new();
    tmp.file("q.fa", include_str!("fixtures/ncbi/query.fa"));
    tmp.file("s.fa", include_str!("fixtures/ncbi/subject.fa"));
    let out = execute(
        SEARCH,
        &[
            "--compatibility",
            "ncbi",
            "--query",
            "q.fa",
            "--subject",
            "s.fa",
            "--backend",
            "cpu",
            "--dust",
            "no",
            "--outfmt",
            "json",
        ],
        &tmp.0,
    );
    success(&out);
    let actual: Vec<serde_json::Value> = serde_json::from_slice(&out.stdout).unwrap();
    let expected: Vec<_> = include_str!("fixtures/ncbi/alignments.tsv")
        .lines()
        .collect();
    assert_eq!(actual.len(), expected.len());
    for (hit, row) in actual.iter().zip(expected) {
        let f: Vec<_> = row.split('\t').collect();
        assert_eq!(hit["qseqid"], f[0]);
        assert_eq!(hit["sseqid"], f[1]);
        for (field, col) in [
            ("length", 3),
            ("mismatch", 4),
            ("gapopen", 5),
            ("qstart", 6),
            ("qend", 7),
            ("sstart", 8),
            ("send", 9),
            ("score", 12),
        ] {
            assert_eq!(
                hit[field],
                f[col].parse::<usize>().unwrap(),
                "{field}: {row}"
            );
        }
        // Decode independent NCBI BTOP into operation runs in output orientation.
        let mut ops = Vec::<u8>::new();
        let btop = f[13].as_bytes();
        let mut i = 0;
        while i < btop.len() {
            if btop[i].is_ascii_digit() {
                let mut n = 0;
                while i < btop.len() && btop[i].is_ascii_digit() {
                    n = n * 10 + (btop[i] - b'0') as usize;
                    i += 1;
                }
                ops.extend(std::iter::repeat_n(b'=', n));
            } else {
                ops.push(if btop[i] == b'-' {
                    b'D'
                } else if btop[i + 1] == b'-' {
                    b'I'
                } else {
                    b'X'
                });
                i += 2;
            }
        }
        let mut cigar = String::new();
        let mut start = 0;
        while start < ops.len() {
            let n = ops[start..]
                .iter()
                .take_while(|&&op| op == ops[start])
                .count();
            cigar.push_str(&format!("{n}{}", ops[start] as char));
            start += n;
        }
        assert_eq!(hit["cigar"], cigar, "{row}");
    }
}

#[test]
fn fasta_database_parallel_and_coordinates() {
    let tmp = Temp::new();
    tmp.file("q.fa", ">q first\nACGTTGCAACCTGATCGTAC\n");
    tmp.file(
        "s.fa",
        ">forward\nTTTTACGTTGCAACCTGATCGTACAAAA\n>reverse\nGTACGATCAGGTTGCAACGT\n",
    );
    success(&execute(
        MAKEDB,
        &["-in", "s.fa", "-out", "subject.rdb", "-dbtype", "nucl"],
        &tmp.0,
    ));
    let a = execute(
        SEARCH,
        &[
            "-query", "q.fa", "-subject", "s.fa", "-dust", "no", "-backend", "scalar",
        ],
        &tmp.0,
    );
    let b = execute(
        SEARCH,
        &[
            "--query",
            "q.fa",
            "--db",
            "subject.rdb",
            "--dust",
            "no",
            "--backend",
            "cpu",
            "--num-threads",
            "4",
        ],
        &tmp.0,
    );
    success(&a);
    success(&b);
    assert_eq!(a.stdout, b.stdout);
    let text = String::from_utf8(a.stdout).unwrap();
    let rows: Vec<Vec<_>> = text.lines().map(|l| l.split('\t').collect()).collect();
    assert_eq!(rows.len(), 2);
    for row in &rows {
        assert_eq!(row.len(), 12);
        assert_eq!(row[2], "100.000");
        assert_eq!(row[3], "20");
        assert_eq!(&row[6..8], &["1", "20"]);
    }
    assert!(
        rows.iter()
            .any(|r| r[1] == "forward" && r[8] == "5" && r[9] == "24")
    );
    assert!(
        rows.iter()
            .any(|r| r[1] == "reverse" && r[8] == "20" && r[9] == "1")
    );
    let extracted = execute(
        DBCMD,
        &["--db", "subject.rdb", "--entry", "reverse"],
        &tmp.0,
    );
    success(&extracted);
    assert_eq!(
        String::from_utf8(extracted.stdout).unwrap(),
        ">reverse\nGTACGATCAGGTTGCAACGT\n"
    );
}

#[test]
fn gap_counts_json_and_no_hit() {
    let tmp = Temp::new();
    tmp.file("q.fa", ">q\nACGTACTGACCTGAAAATCGTACGATCG\n");
    tmp.file("s.fa", ">s\nACGTACTGACCTGTCGTACGATCG\n");
    let out = execute(
        SEARCH,
        &[
            "--query",
            "q.fa",
            "--subject",
            "s.fa",
            "--dust",
            "no",
            "--word-size",
            "7",
            "--outfmt",
            "json",
            "--backend",
            "cpu",
        ],
        &tmp.0,
    );
    success(&out);
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let hit = &json[0];
    assert_eq!(hit["score"], 35);
    assert_eq!(hit["gapopen"], 1);
    assert_eq!(hit["mismatch"], 0);
    assert_eq!(hit["length"], 28);
    assert!(hit["cigar"].as_str().unwrap().contains("4I"));
    let nohit = execute(
        SEARCH,
        &[
            "--query",
            "q.fa",
            "--subject",
            "s.fa",
            "--outfmt",
            "json",
            "--min-score",
            "1000",
            "--backend",
            "cpu",
        ],
        &tmp.0,
    );
    success(&nohit);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&nohit.stdout).unwrap(),
        serde_json::json!([])
    );
}

#[test]
fn invalid_input_and_no_clobber() {
    let tmp = Temp::new();
    tmp.file("q.fa", ">q\nACGTTGCAACCTGATCGTAC\n");
    tmp.file("bad.fa", ">s\nACG\n>s\nTTA\n");
    let out = execute(SEARCH, &["--query", "q.fa", "--subject", "bad.fa"], &tmp.0);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("duplicate"));
    let before = fs::read(tmp.0.join("q.fa")).unwrap();
    let out = execute(
        SEARCH,
        &["--query", "q.fa", "--subject", "q.fa", "--out", "./q.fa"],
        &tmp.0,
    );
    assert!(!out.status.success());
    assert_eq!(fs::read(tmp.0.join("q.fa")).unwrap(), before);
    let out = execute(
        SEARCH,
        &[
            "--query",
            "q.fa",
            "--subject",
            "q.fa",
            "--task",
            "megablast",
        ],
        &tmp.0,
    );
    assert!(!out.status.success());
    let out = execute(MAKEDB, &["--in", "bad.fa", "--out", "bad.rdb"], &tmp.0);
    assert!(!out.status.success());
    assert!(!tmp.0.join("bad.rdb").exists());
}

#[test]
fn query_batches_limits_masking_and_formats_are_identical() {
    let tmp = Temp::new();
    tmp.file("q.fa", ">z\nACGTTGCAACCTGATCGTAC\n>a\nGTACGATCAGGTTGCAACGT\n>lower\nacgttgcaACCTGATCGTAC\n>nohit\nNNNNNNNNNNNNNNNNNNNN\n");
    tmp.file("s.fa", ">z\nACGTTGCAACCTGATCGTACNNACGTTGCAACCTGATCGTAC\n>a\nGTACGATCAGGTTGCAACGT\n>b\nACGTTGCAACCTGATCGTAC\n");
    success(&execute(
        MAKEDB,
        &["--in", "s.fa", "--out", "s.rdb"],
        &tmp.0,
    ));
    for format in ["6", "7", "json"] {
        for dust in ["yes", "no"] {
            let common = [
                "--query",
                "q.fa",
                "--word-size",
                "7",
                "--outfmt",
                format,
                "--dust",
                dust,
                "--lcase-masking",
                "--max-target-seqs",
                "2",
                "--max-hsps",
                "1",
            ];
            let mut scalar = common.to_vec();
            scalar.extend([
                "--backend",
                "scalar",
                "--subject",
                "s.fa",
                "--query-batch-size",
                "1",
            ]);
            let expected = execute(SEARCH, &scalar, &tmp.0);
            success(&expected);
            for (size, bases) in [("2", "1"), ("2", "65536"), ("32", "65536")] {
                let mut args = common.to_vec();
                args.extend([
                    "--backend",
                    "cpu",
                    "--db",
                    "s.rdb",
                    "--num-threads",
                    "4",
                    "--query-batch-size",
                    size,
                    "--query-batch-bases",
                    bases,
                    "--profile",
                ]);
                let actual = execute(SEARCH, &args, &tmp.0);
                success(&actual);
                assert_eq!(
                    actual.stdout, expected.stdout,
                    "format={format} dust={dust} size={size} bases={bases}"
                );
                assert!(String::from_utf8_lossy(&actual.stderr).contains("profile: queries="));
            }
        }
    }
}

#[test]
fn forced_simd_and_subject_batch_controls_preserve_cli_output() {
    let tmp = Temp::new();
    let sequence = "ACGTTGCAACCTGATCGTACTGACGATGCTAGCTTGACCGATTCGTACGTTGCAACCTGATCGTACTGACGATGCTAGCTTGACCGATTCGT";
    tmp.file("q.fa", &format!(">q\n{sequence}\n"));
    tmp.file(
        "s.fa",
        &(0..9)
            .map(|i| format!(">s{i}\n{sequence}\n"))
            .collect::<String>(),
    );
    let common = [
        "--query",
        "q.fa",
        "--subject",
        "s.fa",
        "--dust",
        "no",
        "--outfmt",
        "json",
    ];
    let mut args = common.to_vec();
    args.extend(["--backend", "scalar", "--subject-batch-size", "1"]);
    let expected = execute(SEARCH, &args, &tmp.0);
    success(&expected);
    for (size, bases) in [("1", "1"), ("3", "256"), ("128", "1048576")] {
        let mut args = common.to_vec();
        args.extend([
            "--backend",
            "simd",
            "--num-threads",
            "4",
            "--subject-batch-size",
            size,
            "--subject-batch-bases",
            bases,
        ]);
        let actual = execute(SEARCH, &args, &tmp.0);
        success(&actual);
        assert_eq!(actual.stdout, expected.stdout);
    }
    let mut args = common.to_vec();
    args.extend(["--subject-batch-size", "0"]);
    assert!(!execute(SEARCH, &args, &tmp.0).status.success());
}
