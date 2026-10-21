use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoringScheme {
    pub reward: i32,
    pub penalty: i32,
    pub gap_open: i32,
    pub gap_extend: i32,
}
impl Default for ScoringScheme {
    fn default() -> Self {
        Self {
            reward: 2,
            penalty: -3,
            gap_open: 5,
            gap_extend: 2,
        }
    }
}
impl ScoringScheme {
    pub fn validate(self) -> Result<()> {
        if !(1..=100).contains(&self.reward)
            || !(-100..=-1).contains(&self.penalty)
            || !(0..=100).contains(&self.gap_open)
            || !(1..=100).contains(&self.gap_extend)
        {
            return Err(Error("invalid scoring: reward 1..100, penalty -100..-1, gapopen 0..100, gapextend 1..100".into()));
        }
        Ok(())
    }
    pub fn substitution(self, a: u8, b: u8) -> i32 {
        if a < 4 && a == b {
            self.reward
        } else {
            self.penalty
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlignOptions {
    /// None searches the full subject flank; Some(b) limits |query_offset-subject_offset|.
    pub band: Option<usize>,
    pub xdrop: i32,
    pub max_cells: usize,
    pub scoring: ScoringScheme,
}
impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            band: Some(64),
            xdrop: 50,
            max_cells: 20_000_000,
            scoring: ScoringScheme::default(),
        }
    }
}
impl AlignOptions {
    pub fn validate(self) -> Result<()> {
        self.scoring.validate()?;
        if !(0..=1_000_000).contains(&self.xdrop)
            || self.band.is_some_and(|b| b > 10_000_000)
            || self.max_cells == 0
            || self.max_cells > u32::MAX as usize
        {
            return Err(Error(
                "invalid alignment limits: X-drop 0..1e6, band <=10 Mb, max-cells 1..u32::MAX"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct AlignRequest<'a> {
    pub query: &'a [u8],
    pub subject: &'a [u8],
    pub query_anchor: usize,
    pub subject_anchor: usize,
    pub word_size: usize,
}
impl AlignRequest<'_> {
    pub fn validate(&self) -> Result<()> {
        let qend = self
            .query_anchor
            .checked_add(self.word_size)
            .ok_or_else(|| Error("anchor overflow".into()))?;
        let send = self
            .subject_anchor
            .checked_add(self.word_size)
            .ok_or_else(|| Error("anchor overflow".into()))?;
        if self.word_size == 0
            || qend > self.query.len()
            || send > self.subject.len()
            || self.query[self.query_anchor..qend] != self.subject[self.subject_anchor..send]
            || self.query[self.query_anchor..qend].iter().any(|&b| b > 3)
        {
            return Err(Error(
                "alignment requires an in-bounds exact ACGT seed".into(),
            ));
        }
        if self.query.len().max(self.subject.len()) > 10_000_000 {
            return Err(Error(
                "alignment sequence exceeds the current 10 Mb per-record limit".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AlignmentScore {
    pub score: i32,
    pub query_start: usize,
    pub query_end: usize,
    pub subject_start: usize,
    pub subject_end: usize,
    pub band_edge_reached: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Alignment {
    pub summary: AlignmentScore,
    /// = and X consume both sequences; I consumes query; D consumes subject.
    pub operations: Vec<u8>,
}
impl Alignment {
    pub fn cigar(&self) -> String {
        let mut cigar = String::new();
        let mut start = 0;
        while start < self.operations.len() {
            let mut end = start + 1;
            while end < self.operations.len() && self.operations[end] == self.operations[start] {
                end += 1;
            }
            cigar.push_str(&format!(
                "{}{}",
                end - start,
                self.operations[start] as char
            ));
            start = end;
        }
        cigar
    }
}

/// Backend boundary is a batch, so GPU dispatch never occurs once per DP cell or base.
pub trait GappedAligner: Send + Sync {
    fn name(&self) -> &'static str;
    /// Opt in to cross-query queues with identical AlignOptions. A hint is capped
    /// by search memory limits. On errors search may replay its legacy schedule.
    fn shared_query_batch_size(&self) -> Option<usize> {
        None
    }
    /// Optional cumulative transport telemetry; not elapsed or GPU hardware time.
    fn batch_profile(&self) -> Option<BatchProfile> {
        None
    }
    fn score_batch(
        &self,
        requests: &[AlignRequest<'_>],
        options: AlignOptions,
    ) -> Result<Vec<AlignmentScore>>;
    /// Exact traceback of selected seeds, in input order. Score-only backends retain
    /// the portable CPU implementation; workers supply reusable scalar storage.
    fn traceback_batch(
        &self,
        requests: &[AlignRequest<'_>],
        options: AlignOptions,
        scratch: &mut WorkerScratch,
    ) -> Result<Vec<Alignment>> {
        requests
            .iter()
            .map(|&request| align_seeded_with_scratch(request, options, true, scratch))
            .collect()
    }
}

/// Histogram buckets: 0, 1, 2..3, 4..7, ..., 512..1023, 1024+.
pub const BATCH_HISTOGRAM_BINS: usize = 12;
pub fn batch_histogram_bin(size: usize) -> usize {
    (usize::BITS - size.leading_zeros()).min(BATCH_HISTOGRAM_BINS as u32 - 1) as usize
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchDistribution {
    pub count: usize,
    pub total: usize,
    pub max: usize,
    pub histogram: [usize; BATCH_HISTOGRAM_BINS],
}

/// Snapshot after joining callers is exact; a concurrent snapshot is approximate.
#[derive(Clone, Debug, Default)]
pub struct BatchProfile {
    pub backend: &'static str,
    pub calls: BatchDistribution,
    pub dispatches: BatchDistribution,
    pub query_flanks: BatchDistribution,
    pub subject_bases: usize,
    pub bytes_uploaded: usize,
    pub buffer_allocations: usize,
    pub bind_groups_created: usize,
}

const NEG: i32 = -1_000_000_000;
#[derive(Default)]
struct Side {
    score: i32,
    q: usize,
    s: usize,
    ops: Vec<u8>,
    edge: bool,
}
struct TraceRow {
    start: usize,
    offset: usize,
}

/// Reusable storage owned by one worker. It contains no state shared between alignments.
#[derive(Default)]
pub struct WorkerScratch {
    prev: Vec<i32>,
    curr: Vec<i32>,
    prev_gap: Vec<i32>,
    curr_gap: Vec<i32>,
    trace_rows: Vec<TraceRow>,
    trace_cells: Vec<u8>,
    #[cfg(test)]
    evaluated_cells: usize,
}

#[inline]
fn base<const REVERSE: bool>(sequence: &[u8], position: usize) -> u8 {
    sequence[if REVERSE {
        sequence.len() - 1 - position
    } else {
        position
    }]
}

/// Seed-anchored affine DP, with no local restart. Cost of length L gap = open + L*extend.
/// Rolling score rows and one byte of traceback state per visited cell.
fn extend_side<const REVERSE: bool>(
    query: &[u8],
    subject: &[u8],
    options: AlignOptions,
    traceback: bool,
    scratch: &mut WorkerScratch,
) -> Result<Side> {
    let score = options.scoring;
    let band = options.band.unwrap_or(query.len().max(subject.len()));
    let n = subject.len().min(query.len().saturating_add(band));
    let subject = if REVERSE {
        &subject[subject.len() - n..]
    } else {
        &subject[..n]
    };
    let open = score.gap_open + score.gap_extend;
    let WorkerScratch {
        prev,
        curr,
        prev_gap,
        curr_gap,
        trace_rows: trace,
        trace_cells,
        #[cfg(test)]
        evaluated_cells,
    } = scratch;
    #[cfg(test)]
    {
        *evaluated_cells = 0;
    }
    for row in [&mut *prev, &mut *curr, &mut *prev_gap, &mut *curr_gap] {
        row.resize(n + 1, NEG);
    }
    let mut prev_start = 0;
    // Row zero contains only the origin and affordable leading deletions.
    // Positive extension cost guarantees that no later column can become live.
    let leading = options.xdrop.saturating_sub(score.gap_open).max(0) / score.gap_extend;
    let mut prev_end = n.min(band).min(leading as usize);
    // Only this interval is readable; every subsequent row overwrites its live interval.
    prev[..=prev_end].fill(NEG);
    prev_gap[..=prev_end].fill(NEG);
    prev[0] = 0;
    for (j, h) in prev.iter_mut().enumerate().take(prev_end + 1).skip(1) {
        let value = -score.gap_open - score.gap_extend * j as i32;
        if value >= -options.xdrop {
            *h = value;
        }
    }
    trace.clear();
    trace_cells.clear();
    if traceback {
        trace.push(TraceRow {
            start: 0,
            offset: 0,
        });
        trace_cells.resize(prev_end + 1, 3 | 8);
    }
    let mut best = Side::default();
    let mut visited = 0usize;
    for i in 1..=query.len() {
        let band_start = i.saturating_sub(band);
        let end = n.min(i.saturating_add(band));
        if band_start > end {
            break;
        }
        // Preserve the original geometric cell budget, including the final dead row.
        // Optimizing away arithmetic must not change which requests return an error.
        visited += end - band_start + 1;
        if visited > options.max_cells {
            return Err(Error(
                "alignment cell budget exceeded; increase --max-cells or reduce --band".into(),
            ));
        }
        let trace_offset = trace_cells.len();
        let mut horizontal = NEG;
        let mut left_h = NEG;
        // No local restart: cells before the first surviving predecessor are unreachable.
        // Column zero cannot revive either: its gap cost only grows and best never falls.
        let start = band_start.max(prev_start);
        let mut first_live = usize::MAX;
        let mut last_live = 0;
        for j in start..=end {
            #[cfg(test)]
            {
                *evaluated_cells += 1;
            }
            let mut code;
            let mut vertical;
            let mut h;
            if j == 0 {
                h = -score.gap_open - score.gap_extend * i as i32;
                vertical = h;
                code = 2 | 4;
            } else {
                let above = if j >= prev_start && j <= prev_end {
                    prev[j]
                } else {
                    NEG
                };
                let above_gap = if j >= prev_start && j <= prev_end {
                    prev_gap[j]
                } else {
                    NEG
                };
                let e_open = left_h - open;
                let e_extend = horizontal - score.gap_extend;
                horizontal = e_open.max(e_extend);
                let f_open = above - open;
                let f_extend = above_gap - score.gap_extend;
                vertical = f_open.max(f_extend);
                let diag = if j > prev_start && j - 1 <= prev_end {
                    prev[j - 1]
                } else {
                    NEG
                };
                h = diag
                    + score.substitution(
                        base::<REVERSE>(query, i - 1),
                        base::<REVERSE>(subject, j - 1),
                    );
                code = 1;
                if vertical > h {
                    h = vertical;
                    code = 2;
                }
                if horizontal > h {
                    h = horizontal;
                    code = 3;
                }
                if f_extend > f_open {
                    code |= 4;
                }
                if e_extend > e_open {
                    code |= 8;
                }
            }
            if best.score - h > options.xdrop {
                h = NEG;
                vertical = NEG;
                horizontal = NEG;
                code = 0;
            } else {
                first_live = first_live.min(j);
                last_live = j;
                if options.band.is_some()
                    && i.abs_diff(j) == band
                    && i < query.len()
                    && j < subject.len()
                {
                    best.edge = true;
                }
                if h > best.score {
                    best.score = h;
                    best.q = i;
                    best.s = j;
                }
            }
            curr[j] = h;
            curr_gap[j] = vertical;
            left_h = h;
            if traceback {
                trace_cells.push(code);
            }
            // Keep prev_end + 1: it can still receive a diagonal transition.
            // Beyond it only the horizontal chain remains; once dead it cannot revive.
            if h == NEG && j > prev_end {
                break;
            }
        }
        if traceback {
            trace.push(TraceRow {
                start,
                offset: trace_offset,
            });
        }
        std::mem::swap(prev, curr);
        std::mem::swap(prev_gap, curr_gap);
        if first_live == usize::MAX {
            break;
        }
        // Do not re-prune against the final row best: earlier surviving cells must
        // retain their states under the original running-best X-drop semantics.
        prev_start = first_live;
        prev_end = last_live;
    }
    if traceback {
        let (mut i, mut j, mut state) = (best.q, best.s, 0);
        while i != 0 || j != 0 {
            if i == 0 {
                best.ops.push(b'D');
                j -= 1;
                continue;
            }
            if j == 0 {
                best.ops.push(b'I');
                i -= 1;
                continue;
            }
            let row = &trace[i];
            let code = trace_cells[row.offset + j - row.start];
            if state == 0 {
                state = code & 3;
            }
            match state {
                1 => {
                    best.ops.push(
                        if base::<REVERSE>(query, i - 1) < 4
                            && base::<REVERSE>(query, i - 1) == base::<REVERSE>(subject, j - 1)
                        {
                            b'='
                        } else {
                            b'X'
                        },
                    );
                    i -= 1;
                    j -= 1;
                    state = 0;
                }
                2 => {
                    best.ops.push(b'I');
                    i -= 1;
                    state = if code & 4 != 0 { 2 } else { 0 };
                }
                3 => {
                    best.ops.push(b'D');
                    j -= 1;
                    state = if code & 8 != 0 { 3 } else { 0 };
                }
                _ => return Err(Error("invalid traceback state".into())),
            }
        }
        best.ops.reverse();
    }
    Ok(best)
}

pub fn align_seeded(
    request: AlignRequest<'_>,
    options: AlignOptions,
    traceback: bool,
) -> Result<Alignment> {
    align_seeded_with_scratch(request, options, traceback, &mut WorkerScratch::default())
}

pub fn align_seeded_with_scratch(
    request: AlignRequest<'_>,
    options: AlignOptions,
    traceback: bool,
    scratch: &mut WorkerScratch,
) -> Result<Alignment> {
    request.validate()?;
    options.validate()?;
    let q = request.query_anchor;
    let s = request.subject_anchor;
    let k = request.word_size;
    let sleft_len = options.band.map_or(s, |b| s.min(q.saturating_add(b)));
    let left = extend_side::<true>(
        &request.query[..q],
        &request.subject[s - sleft_len..s],
        options,
        traceback,
        scratch,
    )?;
    let right = extend_side::<false>(
        &request.query[q + k..],
        &request.subject[s + k..],
        options,
        traceback,
        scratch,
    )?;
    let summary = AlignmentScore {
        score: left.score + right.score + k as i32 * options.scoring.reward,
        query_start: q - left.q,
        query_end: q + k + right.q,
        subject_start: s - left.s,
        subject_end: s + k + right.s,
        band_edge_reached: left.edge || right.edge,
    };
    let mut operations = Vec::new();
    if traceback {
        operations.extend(left.ops.iter().rev());
        operations.extend(std::iter::repeat_n(b'=', k));
        operations.extend(right.ops);
    }
    Ok(Alignment {
        summary,
        operations,
    })
}

#[cfg(test)]
#[path = "align/reference.rs"]
mod reference;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dna::PackedDna;

    fn check_reference(
        request: AlignRequest<'_>,
        options: AlignOptions,
        scratch: &mut WorkerScratch,
    ) {
        for traceback in [false, true] {
            assert_eq!(
                align_seeded_with_scratch(request, options, traceback, scratch).map_err(|e| e.0),
                reference::align_seeded(request, options, traceback).map_err(|e| e.0),
                "q={:?} s={:?} anchors={}/{} traceback={traceback} options={options:?}",
                request.query,
                request.subject,
                request.query_anchor,
                request.subject_anchor,
            );
        }
    }

    #[test]
    fn active_frontier_exhaustive_short_flanks_match_frozen_traceback() {
        // Include ambiguity, empty sides, leading gaps, ties, zero X-drop, and both directions.
        let mut words = vec![Vec::new()];
        for len in 1..=3 {
            for mut code in 0..3usize.pow(len) {
                words.push(
                    (0..len)
                        .map(|_| {
                            let base = [0, 1, 4][code % 3];
                            code /= 3;
                            base
                        })
                        .collect(),
                );
            }
        }
        let mut scratch = WorkerScratch::default();
        for q in &words {
            for s in &words {
                // Unique valid seed in the middle tests reverse and forward in one alignment.
                let query: Vec<_> = q
                    .iter()
                    .copied()
                    .chain([2])
                    .chain(q.iter().copied())
                    .collect();
                let subject: Vec<_> = s
                    .iter()
                    .copied()
                    .chain([2])
                    .chain(s.iter().copied())
                    .collect();
                let request = AlignRequest {
                    query: &query,
                    subject: &subject,
                    query_anchor: q.len(),
                    subject_anchor: s.len(),
                    word_size: 1,
                };
                for band in [None, Some(0), Some(1), Some(3)] {
                    for xdrop in [0, 1, 3, 7] {
                        for scoring in [
                            ScoringScheme::default(),
                            ScoringScheme {
                                reward: 1,
                                penalty: -1,
                                gap_open: 0,
                                gap_extend: 1,
                            },
                            ScoringScheme {
                                reward: 100,
                                penalty: -100,
                                gap_open: 100,
                                gap_extend: 100,
                            },
                        ] {
                            check_reference(
                                request,
                                AlignOptions {
                                    band,
                                    xdrop,
                                    scoring,
                                    ..Default::default()
                                },
                                &mut scratch,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn active_frontier_preserves_every_small_budget_boundary() {
        let q = [0; 12];
        let mut scratch = WorkerScratch::default();
        for s in [[0; 16], [1; 16]] {
            for (qa, sa) in [(0, 0), (5, 7), (11, 15)] {
                let mut subject = s;
                subject[sa] = 0;
                for band in [None, Some(0), Some(1), Some(8)] {
                    for xdrop in [0, 7, 50] {
                        for max_cells in 1..=210 {
                            check_reference(
                                AlignRequest {
                                    query: &q,
                                    subject: &subject,
                                    query_anchor: qa,
                                    subject_anchor: sa,
                                    word_size: 1,
                                },
                                AlignOptions {
                                    band,
                                    xdrop,
                                    max_cells,
                                    ..Default::default()
                                },
                                &mut scratch,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn active_frontier_skips_dead_cells_and_compacts_traceback() {
        let q = vec![0; 512];
        let mut scratch = WorkerScratch::default();
        // Reuse storage across wide/empty/tight frontiers and both orientations.
        for xdrop in [1_000_000, 0, 50, 0] {
            let options = AlignOptions {
                xdrop,
                ..Default::default()
            };
            for reverse in [false, true] {
                let side = if reverse {
                    extend_side::<true>(&q, &q, options, true, &mut scratch)
                } else {
                    extend_side::<false>(&q, &q, options, true, &mut scratch)
                }
                .unwrap();
                assert_eq!(side.score, 1024);
                assert_eq!(side.ops, vec![b'='; 512]);
                if xdrop == 0 {
                    assert!(scratch.evaluated_cells <= 3 * q.len());
                    assert!(scratch.trace_cells.len() <= 3 * q.len() + 1);
                }
            }
            let empty = extend_side::<false>(&q, &[], options, true, &mut scratch).unwrap();
            assert_eq!(empty.score, 0);
            assert!(empty.ops.is_empty());
        }
    }

    #[test]
    fn reused_scratch_matches_frozen_reference_including_traceback_and_errors() {
        let mut random = 901u64;
        let mut next = |n: usize| {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            (random >> 32) as usize % n
        };
        let mut scratch = WorkerScratch::default();
        for case in 0..2400 {
            let k = 1 + next(11);
            let n = k + next(90);
            let m = k + next(110);
            let mut q: Vec<_> = (0..n)
                .map(|_| next(if case % 5 == 0 { 15 } else { 4 }) as u8)
                .collect();
            let mut s: Vec<_> = (0..m).map(|_| next(4) as u8).collect();
            if case % 2 == 0 {
                s[..n.min(m)].copy_from_slice(&q[..n.min(m)]);
            }
            let qa = next(n - k + 1);
            let sa = if qa + k <= m && case % 2 == 0 {
                qa
            } else {
                next(m - k + 1)
            };
            for j in 0..k {
                q[qa + j] = next(4) as u8;
                s[sa + j] = q[qa + j];
            }
            let request = AlignRequest {
                query: &q,
                subject: &s,
                query_anchor: qa,
                subject_anchor: sa,
                word_size: k,
            };
            let options = AlignOptions {
                band: [None, Some(0), Some(1), Some(7), Some(64)][case % 5],
                xdrop: [0, 1, 7, 25, 50, 1000][case % 6],
                max_cells: if case % 17 == 0 {
                    1
                } else if case % 19 == 0 {
                    40
                } else {
                    20_000_000
                },
                scoring: ScoringScheme {
                    reward: 1 + next(3) as i32,
                    penalty: -1 - next(5) as i32,
                    gap_open: next(9) as i32,
                    gap_extend: 1 + next(3) as i32,
                },
            };
            for traceback in [false, true] {
                let expected =
                    reference::align_seeded(request, options, traceback).map_err(|e| e.0);
                let actual = align_seeded_with_scratch(request, options, traceback, &mut scratch)
                    .map_err(|e| e.0);
                assert_eq!(
                    actual, expected,
                    "case={case} traceback={traceback} options={options:?}"
                );
            }
        }
    }
    fn dna(s: &[u8]) -> Vec<u8> {
        PackedDna::from_ascii(s).unwrap().unpack()
    }
    fn rescore(alignment: &Alignment, scoring: ScoringScheme) -> i32 {
        let mut prev = 0;
        let mut score = 0;
        for &op in &alignment.operations {
            score += match op {
                b'=' => scoring.reward,
                b'X' => scoring.penalty,
                _ => -scoring.gap_extend - if prev == op { 0 } else { scoring.gap_open },
            };
            prev = op;
        }
        score
    }
    #[test]
    fn insertion_trace_and_score_only_agree() {
        let q = dna(b"ACGTACTGACCTGAAAATCGTACGATCG");
        let s = dna(b"ACGTACTGACCTGTCGTACGATCG");
        let req = AlignRequest {
            query: &q,
            subject: &s,
            query_anchor: 0,
            subject_anchor: 0,
            word_size: 7,
        };
        let a = align_seeded(req, AlignOptions::default(), true).unwrap();
        assert_eq!(a.summary.score, s.len() as i32 * 2 - 5 - 4 * 2);
        assert!(a.cigar().contains("4I"));
        assert_eq!(rescore(&a, ScoringScheme::default()), a.summary.score);
        assert_eq!(
            a.summary,
            align_seeded(req, AlignOptions::default(), false)
                .unwrap()
                .summary
        );
    }
    #[test]
    fn full_matrix_oracle_for_seeded_sides() {
        // Independent full DP oracle without X-drop, compared against banded rolling implementation.
        let scoring = ScoringScheme::default();
        let mut random = 17u64;
        for _ in 0..300 {
            let mut make = |n| {
                (0..n)
                    .map(|_| {
                        random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (random >> 32) as u8 % 4
                    })
                    .collect::<Vec<_>>()
            };
            let q = make(13);
            let s = make(17);
            let (m, n) = (q.len(), s.len());
            let mut h = vec![vec![NEG; n + 1]; m + 1];
            let mut e = h.clone();
            let mut f = h.clone();
            h[0][0] = 0;
            for (i, row) in h.iter_mut().enumerate().skip(1) {
                row[0] = -5 - 2 * i as i32;
            }
            for (j, value) in h[0].iter_mut().enumerate().skip(1) {
                *value = -5 - 2 * j as i32;
            }
            let mut best = 0;
            for i in 1..=m {
                for j in 1..=n {
                    e[i][j] = (h[i][j - 1] - 7).max(e[i][j - 1] - 2);
                    f[i][j] = (h[i - 1][j] - 7).max(f[i - 1][j] - 2);
                    h[i][j] = (h[i - 1][j - 1] + scoring.substitution(q[i - 1], s[j - 1]))
                        .max(e[i][j])
                        .max(f[i][j]);
                    best = best.max(h[i][j]);
                }
            }
            let side = extend_side::<false>(
                &q,
                &s,
                AlignOptions {
                    band: None,
                    xdrop: 1000,
                    ..Default::default()
                },
                true,
                &mut WorkerScratch::default(),
            )
            .unwrap();
            assert_eq!(side.score, best);
            let a = Alignment {
                summary: AlignmentScore {
                    score: side.score,
                    query_start: 0,
                    query_end: side.q,
                    subject_start: 0,
                    subject_end: side.s,
                    band_edge_reached: false,
                },
                operations: side.ops,
            };
            assert_eq!(rescore(&a, scoring), best);
            assert_eq!(a.operations.iter().filter(|&&b| b != b'D').count(), side.q);
            assert_eq!(a.operations.iter().filter(|&&b| b != b'I').count(), side.s);
        }
    }
    #[test]
    fn rejects_invalid_seeds_and_budget() {
        let seq = vec![0; 100];
        let req = AlignRequest {
            query: &seq,
            subject: &seq,
            query_anchor: 0,
            subject_anchor: 0,
            word_size: 5,
        };
        assert!(
            align_seeded(
                req,
                AlignOptions {
                    max_cells: 1,
                    ..Default::default()
                },
                false
            )
            .is_err()
        );
        assert!(
            align_seeded(
                AlignRequest {
                    word_size: 101,
                    ..req
                },
                AlignOptions::default(),
                false
            )
            .is_err()
        );
    }
}
