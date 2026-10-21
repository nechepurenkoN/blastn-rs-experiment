# Independent NCBI BLAST+ fixtures

Generated with the official native ARM64 **BLAST+ 2.17.0**, build Jul 1 2025,
not with `blastn-rs`. The deterministic FASTA covers an exact match, reverse
strand, insertion, duplicate-score subjects in non-lexical database order,
multiple HSPs, a long homopolymer and a short query whose cutoff must be
limited by its effective search space. `no.tsv` deliberately retains two
contained off-diagonal HSPs separated by exactly 50 diagonals.

Common reference command (run from workspace root `blast-rs`):

```sh
BLAST_USAGE_REPORT=false /path/to/ncbi/bin/blastn \
  -task blastn \
  -query crates/blast-cli/tests/fixtures/ncbi/query.fa \
  -subject crates/blast-cli/tests/fixtures/ncbi/subject.fa \
  -xdrop_ungap 20 -xdrop_gap 50 -xdrop_gap_final 50 \
  -no_greedy -window_size 0 -max_target_seqs 100000 \
  -dust no -outfmt 6 -out /path/to/new-output.tsv
```

Variants:

| File | Changes to common command |
|---|---|
| `no.tsv` | none |
| `yes.tsv` | `-dust yes` |
| `one.tsv` | `-max_target_seqs 1` |
| `score13.tsv` | `-dust yes -reward 1 -penalty -3 -gapopen 2 -gapextend 2` |
| `score13-gap21.tsv` | `-dust yes -reward 1 -penalty -3 -gapopen 2 -gapextend 1` |
| `minus.tsv` | `-dust yes -strand minus` |
| `alignments.tsv` | `-outfmt '6 std score btop'` |

End-to-end tests compare the TSV bytes, not unordered coordinate sets. The BTOP
fixture independently checks raw score, operation order, gap orientation and
CIGAR in the Rust-specific JSON format. NCBI is not required during `cargo test`.

`score13-gap21.tsv` was generated on 2026-10-17 with the same official 2.17.0
ARM64 executable. It covers the alpha=1.2 length correction for gap costs 2/1;
the older 2/2 fixture does not exercise that table row. A separate CLI regression
test checks an exact 20-base hit at E-value 5e-5 in both native/ncbi profiles.

`blast-core/src/mask/ncbi-2.17.0.tsv` contains 128 independently generated SDUST
cases (name, ACGT sequence, comma-separated inclusive zero-based intervals).
The source is NCBI `-dust yes -outfmt 15` → `query_masking`, with the same subject
FASTA. Every query has an unmasked exact anchor and **at least one reference
hit**: NCBI JSON omits `query_masking` when there are no hits, so treating a
missing field in a no-hit result as an empty mask would produce a false oracle.
Cases cover tandem repeats, window boundaries, random DNA and reverse-oriented
inserts. Inputs themselves are saved in that TSV, so regeneration needs no RNG.

Full production-data comparisons are separate: `compare_ncbi --compatibility
ncbi --require-identical`. See `docs/NCBI_COMPATIBILITY.md` for limits and results.
