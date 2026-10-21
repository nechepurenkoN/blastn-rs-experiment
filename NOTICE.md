# NCBI attribution

This project studies the nucleotide BLAST algorithms implemented by the National
Center for Biotechnology Information (NCBI), National Library of Medicine, NIH.
Statistical constants in `crates/blast-core/src/stats.rs` are transcribed from
NCBI C++ Toolkit `src/algo/blast/core/blast_stat.c` (`blastn_values_2_3` and
`blastn_values_1_3`). The effective-length calculation follows the fixed-point
inequality and upper bound documented in `BLAST_ComputeLengthAdjustment`.

The batch seed lookup in `crates/blast-core/src/seed/lookup.rs` and reusable
diagonal tracking in `crates/blast-core/src/search/diagonal.rs` adapt the
backbone/presence-vector and diagonal-array techniques studied in NCBI's
`blast_nalookup.h`, `blast_nascan.c`, and `na_ungapped.c`. These are Rust
implementations of the techniques, not line-for-line translations: full-word
verification, stride-one scan order, tagged ring slots, and the existing Rust
one-hit/two-hit behavior are retained. NCBI cutoffs and off-diagonal two-hit
heuristics have not been adopted in this stage.

The active DP frontiers in `crates/blast-core/src/align.rs`,
`crates/blast-cpu/src/neon_dp.rs`, and `crates/blast-gpu/src/extend.wgsl`
adapt the first/last-live-column technique
studied in `src/algo/blast/core/blast_gapalign.c`. They are independently
implemented for the existing Rust recurrence, running-best X-drop and tie
order. The NEON kernel uses a conservative union of four lane frontiers;
the GPU kernel maintains a separate frontier per seed flank and initializes
only readable storage intervals.
NCBI's minimum X-drop adjustment, recurrence/state layout, and preliminary
versus final extension policies are not copied.

The opt-in `--compatibility ncbi` profile additionally adapts ungapped
Karlin statistics (`BlastKarlinLHtoK`), gap-trigger/cutoff conversion
(`BlastInitialWordParametersUpdate`), HSP/subject ordering and containment
(`blast_hits.c`, `blast_itree.c`, `blast_nucl_options.cpp`), and tabular
number formatting (`CAlignFormatUtil::GetScoreString`, `CBlastTabularInfo::SetScores`).
It applies the minimum gapped X-drop adjustment; the native profile still
retains its previous policy. The full preliminary/final NCBI pipeline is not
ported. The symmetric DUST implementation follows Aleksandr Morgulis's
perfect-interval algorithm in `include/algo/dustmask/sdust.tex` and
`src/algo/dustmask/symdust.cpp`, using a simpler bounded-window implementation.
NCBI's ambiguous-base random converter is not implemented.
Frozen conformance fixtures were generated independently by NCBI BLAST+ 2.17.0.

Reference source: https://github.com/ncbi/ncbi-cxx-toolkit-public
Commit examined: `712028344fa566b9886fe0041a071f9ed24751cf`.

NCBI-authored source is a United States Government Work, freely available for use
and reproduction. NCBI asks users to cite the authors in products based on this
material. The NLM and U.S. Government disclaim warranties of performance,
merchantability or fitness for any particular purpose. The upstream repository
contains third-party components with separate licenses; they were not copied
into this implementation. See the upstream LICENSE for the full notice.

References:

* Altschul et al. (1990), Basic local alignment search tool, J. Mol. Biol. 215:403–410.
* Altschul et al. (1997), Gapped BLAST and PSI-BLAST, Nucleic Acids Res. 25:3389–3402.
* Camacho et al. (2009), BLAST+: architecture and applications, BMC Bioinformatics 10:421.

This implementation is not an NCBI product or an assertion of NCBI endorsement.
