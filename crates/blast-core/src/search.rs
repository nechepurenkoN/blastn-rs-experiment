use crate::{
    Error, Result,
    align::{
        AlignOptions, AlignRequest, Alignment, AlignmentScore, GappedAligner, WorkerScratch,
        align_seeded_with_scratch,
    },
    dna::{SequenceRecord, SequenceView, reverse_complement},
    mask::{dust_mask, symmetric_dust_mask},
    seed::{
        Kmers, QueryLookup,
        lookup::{WordLookup, WordRange},
    },
    stats::{KarlinStats, UngappedStats},
};
use std::{
    collections::{BTreeMap, HashSet},
    ops::Range,
    sync::Arc,
};

mod diagonal;
pub use diagonal::SeedScratch;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Strand {
    Plus,
    Minus,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrandSelection {
    Both,
    Plus,
    Minus,
}

#[derive(Clone, Debug)]
pub struct SearchOptions {
    /// NCBI-compatible cutoffs and subject/HSP ordering; not a full BLAST+ replacement.
    pub ncbi_compatibility: bool,
    pub word_size: usize,
    pub strand: StrandSelection,
    pub dust: bool,
    pub lowercase_masking: bool,
    /// Zero keeps one-hit triggering, as in NCBI blastn defaults.
    pub two_hit_window: usize,
    /// Zero disables frequency suppression (sensitivity-preserving default).
    pub max_seed_occurrences: usize,
    pub ungapped_xdrop: i32,
    pub min_ungapped_score: i32,
    pub min_score: i32,
    pub evalue: f64,
    pub max_candidates: usize,
    pub align: AlignOptions,
}
impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            ncbi_compatibility: false,
            word_size: 11,
            strand: StrandSelection::Both,
            dust: true,
            lowercase_masking: false,
            two_hit_window: 0,
            max_seed_occurrences: 0,
            ungapped_xdrop: 23,
            min_ungapped_score: 16,
            min_score: 0,
            evalue: 10.0,
            max_candidates: 100_000,
            align: AlignOptions::default(),
        }
    }
}
impl SearchOptions {
    pub fn validate(&self) -> Result<()> {
        if !(1..=32).contains(&self.word_size)
            || self.ungapped_xdrop < 0
            || self.align.xdrop < 0
            || self.min_score < 0
            || self.min_ungapped_score < 0
            || self.max_candidates == 0
            || self.align.max_cells == 0
            || !self.evalue.is_finite()
            || self.evalue < 0.0
        {
            return Err(Error("invalid search parameters (word size 1..32; thresholds nonnegative; budgets positive)".into()));
        }
        self.align.validate()?;
        KarlinStats::for_scoring(self.align.scoring)?;
        Ok(())
    }
}

pub struct QueryContext {
    pub sequence: Vec<u8>,
    pub strand: Strand,
    pub lookup: QueryLookup,
}
pub struct PreparedQuery {
    pub id: String,
    pub len: usize,
    pub contexts: Vec<QueryContext>,
    pub options: SearchOptions,
    pub statistics: KarlinStats,
    pub search_space: f64,
}
impl PreparedQuery {
    pub fn new(
        query: &SequenceRecord,
        mut options: SearchOptions,
        db_len: u64,
        db_count: usize,
    ) -> Result<Self> {
        options.validate()?;
        if query.sequence.len() > 10_000_000 {
            return Err(Error("query exceeds 10 Mb per-record limit".into()));
        }
        let statistics = KarlinStats::for_scoring(options.align.scoring)?;
        let search_space = statistics.effective_space(query.sequence.len(), db_len, db_count);
        if options.ncbi_compatibility {
            options.min_ungapped_score = UngappedStats::for_scoring(options.align.scoring)?
                .gap_trigger(27.0)
                .min(statistics.score_cutoff(options.evalue, search_space));
        }
        let mut contexts = Vec::new();
        for strand in [Strand::Plus, Strand::Minus] {
            if (strand == Strand::Plus && options.strand == StrandSelection::Minus)
                || (strand == Strand::Minus && options.strand == StrandSelection::Plus)
            {
                continue;
            }
            let mut sequence = query.sequence.unpack();
            let mut lower = query.sequence.lowercase_mask().to_vec();
            if strand == Strand::Minus {
                sequence = reverse_complement(&sequence);
                lower.reverse();
            }
            let mut mask = if options.dust && options.ncbi_compatibility {
                symmetric_dust_mask(&sequence)?
            } else if options.dust {
                dust_mask(&sequence, 64, 20.0)
            } else {
                vec![false; sequence.len()]
            };
            if options.lowercase_masking {
                for (m, l) in mask.iter_mut().zip(lower) {
                    *m |= l;
                }
            }
            let lookup = QueryLookup::build(
                &sequence,
                &mask,
                options.word_size,
                options.max_seed_occurrences,
            )?;
            contexts.push(QueryContext {
                sequence,
                strand,
                lookup,
            });
        }
        Ok(Self {
            id: query.id.clone(),
            len: query.sequence.len(),
            contexts,
            search_space,
            statistics,
            options,
        })
    }
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct SearchDiagnostics {
    pub seed_hits: u64,
    pub ungapped_extensions: u64,
    pub gapped_candidates: u64,
    pub tracebacks: u64,
    pub band_edge_candidates: u64,
}
impl SearchDiagnostics {
    pub fn merge(&mut self, other: &Self) {
        self.seed_hits += other.seed_hits;
        self.ungapped_extensions += other.ungapped_extensions;
        self.gapped_candidates += other.gapped_candidates;
        self.tracebacks += other.tracebacks;
        self.band_edge_candidates += other.band_edge_candidates;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hsp {
    pub query_id: String,
    pub subject_id: String,
    pub query_len: usize,
    pub subject_len: usize,
    pub strand: Strand,
    pub alignment: Alignment,
    pub bit_score: f64,
    pub evalue: f64,
}
impl Hsp {
    /// BLAST-style coordinates: query ascending; subject descending for reverse hits.
    pub fn coordinates(&self) -> (usize, usize, usize, usize) {
        coordinates(&self.alignment.summary, self.strand, self.query_len)
    }
    pub fn counts(&self) -> (usize, usize, usize) {
        let (mut matches, mut mismatches, mut gapopens) = (0, 0, 0);
        let mut prev = 0;
        for &op in &self.alignment.operations {
            match op {
                b'=' => matches += 1,
                b'X' => mismatches += 1,
                _ if prev != op => gapopens += 1,
                _ => {}
            }
            prev = op;
        }
        (matches, mismatches, gapopens)
    }
}

fn coordinates(
    a: &AlignmentScore,
    strand: Strand,
    query_len: usize,
) -> (usize, usize, usize, usize) {
    match strand {
        Strand::Plus => (
            a.query_start + 1,
            a.query_end,
            a.subject_start + 1,
            a.subject_end,
        ),
        Strand::Minus => (
            query_len - a.query_end + 1,
            query_len - a.query_start,
            a.subject_end,
            a.subject_start + 1,
        ),
    }
}

/// Selection metadata retains the original seed, so a later traceback has identical ties.
#[derive(Clone, Debug)]
pub struct ScoredHsp {
    pub subject_id: Arc<str>,
    pub subject_len: usize,
    pub query_len: usize,
    pub strand: Strand,
    pub summary: AlignmentScore,
    pub bit_score: f64,
    pub evalue: f64,
    query_anchor: usize,
    subject_anchor: usize,
    context: usize,
}
impl ScoredHsp {
    pub fn compare_ncbi_significance(&self, other: &Self) -> std::cmp::Ordering {
        let ev = if self.evalue < 1e-180 && other.evalue < 1e-180 {
            std::cmp::Ordering::Equal
        } else {
            self.evalue.total_cmp(&other.evalue)
        };
        ev.then(other.summary.score.cmp(&self.summary.score))
    }
    /// NCBI s_EvalueCompareHSPs / ScoreCompareHSPs. Coordinates are internal,
    /// including reverse-complement query offsets, not exported coordinates.
    pub fn compare_ncbi(&self, other: &Self) -> std::cmp::Ordering {
        let (a, b) = (&self.summary, &other.summary);
        self.compare_ncbi_significance(other)
            .then(a.subject_start.cmp(&b.subject_start))
            .then(b.subject_end.cmp(&a.subject_end))
            .then(a.query_start.cmp(&b.query_start))
            .then(b.query_end.cmp(&a.query_end))
            .then(self.strand.cmp(&other.strand))
    }
    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        self.evalue
            .total_cmp(&other.evalue)
            .then(other.summary.score.cmp(&self.summary.score))
            .then(self.subject_id.cmp(&other.subject_id))
            .then(self.strand.cmp(&other.strand))
            .then(
                coordinates(&self.summary, self.strand, self.query_len).cmp(&coordinates(
                    &other.summary,
                    other.strand,
                    other.query_len,
                )),
            )
    }

    pub fn traceback(
        &self,
        query: &PreparedQuery,
        subject: SequenceView<'_>,
        scratch: &mut WorkerScratch,
    ) -> Result<Hsp> {
        let alignment = align_seeded_with_scratch(
            self.traceback_request(query, subject)?,
            query.options.align,
            true,
            scratch,
        )?;
        self.finish_traceback(query, alignment)
    }

    /// Recreate the original seed and strand context; never re-anchor a selected HSP.
    pub fn traceback_request<'a>(
        &self,
        query: &'a PreparedQuery,
        subject: SequenceView<'a>,
    ) -> Result<AlignRequest<'a>> {
        if self.subject_id.as_ref() != subject.id || self.subject_len != subject.sequence.len() {
            return Err(Error(
                "traceback subject differs from scored subject".into(),
            ));
        }
        Ok(AlignRequest {
            query: &query.contexts[self.context].sequence,
            subject: subject.sequence,
            query_anchor: self.query_anchor,
            subject_anchor: self.subject_anchor,
            word_size: query.options.word_size,
        })
    }

    /// The selected score, endpoints and band diagnostic must survive traceback exactly.
    pub fn finish_traceback(&self, query: &PreparedQuery, alignment: Alignment) -> Result<Hsp> {
        if alignment.summary != self.summary {
            return Err(Error(
                "backend score/endpoints differ from CPU traceback".into(),
            ));
        }
        Ok(Hsp {
            query_id: query.id.clone(),
            subject_id: self.subject_id.to_string(),
            query_len: self.query_len,
            subject_len: self.subject_len,
            strand: self.strand,
            alignment,
            bit_score: self.bit_score,
            evalue: self.evalue,
        })
    }
}

/// Zero means unlimited. Applied only after all subjects have been scored and ranked.
#[derive(Clone, Copy, Default)]
pub struct HitLimits {
    pub max_target_seqs: usize,
    pub max_hsps: usize,
}

#[derive(Clone, Copy)]
struct Candidate {
    q: usize,
    s: usize,
}
pub type MatchRun = fn(&[u8], &[u8]) -> usize;
pub fn scalar_match_run(q: &[u8], s: &[u8]) -> usize {
    q.iter()
        .zip(s)
        .take_while(|(a, b)| **a < 4 && a == b)
        .count()
}

fn ungapped(
    query: &[u8],
    subject: &[u8],
    q: usize,
    s: usize,
    options: &SearchOptions,
    match_run: MatchRun,
) -> (i32, Range<usize>) {
    let scoring = options.align.scoring;
    let k = options.word_size;
    let (mut running, mut left_best, mut best_left) = (0, 0, 0);
    for step in 1..=q.min(s) {
        running += scoring.substitution(query[q - step], subject[s - step]);
        if running > left_best {
            left_best = running;
            best_left = step;
        }
        if left_best - running > options.ungapped_xdrop {
            break;
        }
    }
    let (mut pos, mut running, mut right_best, mut best_right) = (0, 0, 0, 0);
    let max = (query.len() - q - k).min(subject.len() - s - k);
    while pos < max {
        let run = match_run(
            &query[q + k + pos..q + k + max],
            &subject[s + k + pos..s + k + max],
        );
        pos += run;
        running += run as i32 * scoring.reward;
        if running > right_best {
            right_best = running;
            best_right = pos;
        }
        if pos == max {
            break;
        }
        running += scoring.substitution(query[q + k + pos], subject[s + k + pos]);
        pos += 1;
        if right_best - running > options.ungapped_xdrop {
            break;
        }
    }
    (
        k as i32 * scoring.reward + left_best + right_best,
        s - best_left..s + k + best_right,
    )
}

/// Shared query-side lookup. Each posting still belongs to one independent context.
pub struct QueryBatch<'a> {
    queries: &'a [PreparedQuery],
    contexts: Vec<(usize, usize)>,
    groups: Vec<BatchLookup>,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Posting {
    stream: usize,
    query_position: usize,
}
struct BatchLookup {
    word_size: usize,
    lookup: WordLookup,
    postings: Vec<Posting>,
}
#[derive(Default)]
struct ContextScores {
    seen: HashSet<AlignmentScore>,
    hits: Vec<ScoredHsp>,
}

struct SubjectScores {
    id: Arc<str>,
    len: usize,
    contexts: Vec<ContextScores>,
    diagnostics: Vec<SearchDiagnostics>,
}
struct PendingCandidate {
    subject: usize,
    stream: usize,
    anchor: Candidate,
}
pub type ScoredQueryResults = Vec<(Vec<ScoredHsp>, SearchDiagnostics)>;

impl<'a> QueryBatch<'a> {
    pub fn new(queries: &'a [PreparedQuery]) -> Self {
        let mut contexts = Vec::new();
        let mut words = BTreeMap::<usize, Vec<(u64, Posting)>>::new();
        for (qi, query) in queries.iter().enumerate() {
            for (ci, context) in query.contexts.iter().enumerate() {
                let stream = contexts.len();
                contexts.push((qi, ci));
                let entries = words.entry(query.options.word_size).or_default();
                for (key, positions) in context.lookup.entries() {
                    entries.extend(positions.iter().map(|&query_position| {
                        (
                            key,
                            Posting {
                                stream,
                                query_position,
                            },
                        )
                    }));
                }
            }
        }
        let groups = words
            .into_iter()
            .map(|(word_size, mut words)| {
                // Index layout must never decide which equal-scoring seed wins.
                words.sort_unstable();
                let mut ranges = Vec::new();
                let mut postings = Vec::with_capacity(words.len());
                let mut i = 0;
                while i < words.len() {
                    let mut end = i + 1;
                    while end < words.len() && words[end].0 == words[i].0 {
                        end += 1;
                    }
                    ranges.push(WordRange {
                        word: words[i].0,
                        positions: i..end,
                    });
                    postings.extend(words[i..end].iter().map(|w| w.1));
                    i = end;
                }
                BatchLookup {
                    word_size,
                    lookup: WordLookup::new(word_size, ranges),
                    postings,
                }
            })
            .collect();
        Self {
            queries,
            contexts,
            groups,
        }
    }

    /// One subject scan per distinct word size, regardless of query/strand count.
    pub fn score_subject(
        &self,
        subject: SequenceView<'_>,
        backend: &dyn GappedAligner,
        match_run: MatchRun,
    ) -> Result<ScoredQueryResults> {
        Ok(self
            .score_subjects(&[subject], backend, match_run)?
            .pop()
            .unwrap())
    }

    /// Bounded pending queues combine independent candidates across subjects and strands.
    /// Input sequences must remain alive until every score batch has completed.
    pub fn score_subjects(
        &self,
        subjects: &[SequenceView<'_>],
        backend: &dyn GappedAligner,
        match_run: MatchRun,
    ) -> Result<Vec<ScoredQueryResults>> {
        self.score_subjects_with_scratch(subjects, backend, match_run, &mut SeedScratch::default())
    }

    /// Reuse diagonal storage across worker blocks; no shared mutable search state.
    pub fn score_subjects_with_scratch(
        &self,
        subjects: &[SequenceView<'_>],
        backend: &dyn GappedAligner,
        match_run: MatchRun,
        scratch: &mut SeedScratch,
    ) -> Result<Vec<ScoredQueryResults>> {
        let Some(shared_size) = backend.shared_query_batch_size() else {
            return self.score_subjects_impl::<false>(subjects, backend, match_run, scratch, 256);
        };
        match self.score_subjects_impl::<true>(subjects, backend, match_run, scratch, shared_size) {
            // Reordering flushes must not change the first deterministic failure
            // (for example, candidate budget versus DP budget). Discard partial
            // results and replay the old queues on the SAME backend, not a CPU fallback.
            Err(_) => self.score_subjects_impl::<false>(subjects, backend, match_run, scratch, 256),
            result => result,
        }
    }

    fn score_subjects_impl<const SHARED: bool>(
        &self,
        subjects: &[SequenceView<'_>],
        backend: &dyn GappedAligner,
        match_run: MatchRun,
        scratch: &mut SeedScratch,
        shared_size: usize,
    ) -> Result<Vec<ScoredQueryResults>> {
        scratch.configure(self.contexts.iter().map(|&(qi, ci)| {
            self.queries[qi].contexts[ci]
                .sequence
                .len()
                .saturating_sub(self.queries[qi].options.word_size)
                .saturating_add(1)
        }));
        let mut scored: Vec<_> = subjects
            .iter()
            .map(|s| SubjectScores {
                id: s.id.into(),
                len: s.sequence.len(),
                contexts: self
                    .contexts
                    .iter()
                    .map(|_| ContextScores::default())
                    .collect(),
                diagnostics: vec![SearchDiagnostics::default(); self.queries.len()],
            })
            .collect();
        // Preserve per-query queues for CPU/third-party backends unless they opt in.
        // Only AlignOptions must match: seeds, thresholds, statistics, budgets and
        // result state remain independent per query/subject/context.
        let mut owners = Vec::<usize>::new();
        let mut members = Vec::<usize>::new();
        let destinations: Vec<_> = (0..if SHARED { self.queries.len() } else { 0 })
            .map(|qi| {
                let existing = owners.iter().position(|&other| {
                    self.queries[other].options.align == self.queries[qi].options.align
                });
                if let Some(group) = existing {
                    members[group] += 1;
                    group
                } else {
                    owners.push(qi);
                    members.push(1);
                    owners.len() - 1
                }
            })
            .collect();
        // Never retain more candidates than the old 256-per-query aggregate cap.
        let capacities: Vec<_> = members
            .iter()
            .map(|&n| shared_size.clamp(1, 1024).min(n.saturating_mul(256)))
            .collect();
        let mut pending: Vec<Vec<PendingCandidate>> = if SHARED {
            // Avoid geometric growth past a non-power-of-two aggregate limit.
            capacities
                .iter()
                .map(|&capacity| Vec::with_capacity(capacity))
                .collect()
        } else {
            self.queries.iter().map(|_| Vec::new()).collect()
        };
        for (si, &subject) in subjects.iter().enumerate() {
            let diagnostics =
                self.visit_candidates(subject, match_run, scratch, |stream, anchor| {
                    let qi = self.contexts[stream].0;
                    let group = if SHARED { destinations[qi] } else { qi };
                    pending[group].push(PendingCandidate {
                        subject: si,
                        stream,
                        anchor,
                    });
                    if pending[group].len() == if SHARED { capacities[group] } else { 256 } {
                        self.score_pending::<SHARED>(
                            if SHARED { owners[group] } else { qi },
                            subjects,
                            &mut pending[group],
                            &mut scored,
                            backend,
                        )?;
                    }
                    Ok(())
                })?;
            for (total, found) in scored[si].diagnostics.iter_mut().zip(diagnostics) {
                total.merge(&found);
            }
        }
        for (group, queue) in pending.iter_mut().enumerate() {
            self.score_pending::<SHARED>(
                if SHARED { owners[group] } else { group },
                subjects,
                queue,
                &mut scored,
                backend,
            )?;
        }
        Ok(scored
            .into_iter()
            .map(|subject| {
                let mut output: ScoredQueryResults = subject
                    .diagnostics
                    .into_iter()
                    .map(|d| (Vec::new(), d))
                    .collect();
                // Canonical context/discovery order is independent of scoring batch order.
                for ((qi, _), state) in self.contexts.iter().zip(subject.contexts) {
                    output[*qi].0.extend(state.hits);
                }
                for (qi, (hits, _)) in output.iter_mut().enumerate() {
                    if self.queries[qi].options.ncbi_compatibility {
                        retain_distinct_with_diagonals(hits, true);
                    } else {
                        retain_distinct(hits);
                    }
                }
                output
            })
            .collect())
    }

    fn visit_candidates(
        &self,
        subject: SequenceView<'_>,
        match_run: MatchRun,
        scratch: &mut SeedScratch,
        mut visitor: impl FnMut(usize, Candidate) -> Result<()>,
    ) -> Result<Vec<SearchDiagnostics>> {
        for tracker in &mut scratch.diagonals {
            tracker.next_subject();
        }
        let mut output = vec![SearchDiagnostics::default(); self.queries.len()];
        for group in &self.groups {
            for (s, key) in Kmers::new(subject.sequence, None, group.word_size)? {
                let Some(range) = group.lookup.get(key) else {
                    continue;
                };
                for posting in &group.postings[range] {
                    let (qi, ci) = self.contexts[posting.stream];
                    let prepared = &self.queries[qi];
                    let context = &prepared.contexts[ci];
                    let options = &prepared.options;
                    let diagnostics = &mut output[qi];
                    let q = posting.query_position;
                    diagnostics.seed_hits += 1;
                    let diagonal = scratch.diagonals[posting.stream].entry(s, q);
                    if s < diagonal.covered_until {
                        continue;
                    }
                    if options.two_hit_window > 0 {
                        let previous = diagonal.last_hit();
                        if let Some(p) = previous
                            && s - p < options.word_size
                        {
                            continue;
                        }
                        diagonal.set_last_hit(s);
                        if previous.is_none_or(|p| s - p > options.two_hit_window) {
                            continue;
                        }
                    }
                    diagnostics.ungapped_extensions += 1;
                    let (score, range) = ungapped(
                        &context.sequence,
                        subject.sequence,
                        q,
                        s,
                        options,
                        match_run,
                    );
                    diagonal.covered_until = range.end;
                    if score < options.min_ungapped_score {
                        continue;
                    }
                    diagnostics.gapped_candidates += 1;
                    if diagnostics.gapped_candidates > options.max_candidates as u64 {
                        return Err(Error(format!(
                            "candidate budget exceeded for {} vs {}; increase --max-candidates or enable masking",
                            prepared.id, subject.id,
                        )));
                    }
                    visitor(posting.stream, Candidate { q, s })?;
                }
            }
        }
        Ok(output)
    }

    fn score_pending<const SHARED: bool>(
        &self,
        qi: usize,
        subjects: &[SequenceView<'_>],
        pending: &mut Vec<PendingCandidate>,
        scored: &mut [SubjectScores],
        backend: &dyn GappedAligner,
    ) -> Result<()> {
        if pending.is_empty() {
            return Ok(());
        }
        let owner = &self.queries[qi];
        let options = owner.options.align;
        let requests: Vec<_> = pending
            .iter()
            .map(|c| {
                let (qi, ci) = self.contexts[c.stream];
                let prepared = if SHARED { &self.queries[qi] } else { owner };
                debug_assert_eq!(prepared.options.align, options);
                AlignRequest {
                    query: &prepared.contexts[ci].sequence,
                    subject: subjects[c.subject].sequence,
                    query_anchor: c.anchor.q,
                    subject_anchor: c.anchor.s,
                    word_size: prepared.options.word_size,
                }
            })
            .collect();
        let scores = backend.score_batch(&requests, options)?;
        if scores.len() != pending.len() {
            return Err(Error("backend returned wrong batch length".into()));
        }
        for (candidate, score) in pending.drain(..).zip(scores) {
            let (candidate_qi, context) = self.contexts[candidate.stream];
            let qi = if SHARED { candidate_qi } else { qi };
            let prepared = if SHARED { &self.queries[qi] } else { owner };
            let subject = &mut scored[candidate.subject];
            subject.diagnostics[qi].band_edge_candidates += u64::from(score.band_edge_reached);
            let evalue = prepared
                .statistics
                .evalue(score.score, prepared.search_space);
            if score.score < prepared.options.min_score || evalue > prepared.options.evalue {
                continue;
            }
            let state = &mut subject.contexts[candidate.stream];
            if !state.seen.insert(score.clone()) {
                continue;
            }
            state.hits.push(ScoredHsp {
                subject_id: Arc::clone(&subject.id),
                subject_len: subject.len,
                query_len: prepared.len,
                strand: prepared.contexts[context].strand,
                bit_score: prepared.statistics.bits(score.score),
                evalue,
                summary: score,
                query_anchor: candidate.anchor.q,
                subject_anchor: candidate.anchor.s,
                context,
            });
        }
        Ok(())
    }
}

fn retain_distinct(hits: &mut Vec<ScoredHsp>) {
    retain_distinct_with_diagonals(hits, false);
}

fn retain_distinct_with_diagonals(hits: &mut Vec<ScoredHsp>, ncbi: bool) {
    hits.sort_by(if ncbi {
        ScoredHsp::compare_ncbi
    } else {
        ScoredHsp::compare
    });
    let mut kept: Vec<ScoredHsp> = Vec::new();
    for hsp in hits.drain(..) {
        let a = &hsp.summary;
        if kept.iter().any(|h| {
            let b = &h.summary;
            h.strand == hsp.strand
                && b.query_start <= a.query_start
                && b.query_end >= a.query_end
                && b.subject_start <= a.subject_start
                && b.subject_end >= a.subject_end
                && b.score >= a.score
                && (!ncbi || {
                    // BLASTN SetHitSavingOptionsDefaults uses 50, and
                    // s_HSPIsContained checks either endpoint diagonal.
                    let diagonal = |q: usize, s: usize| q as i64 - s as i64;
                    (diagonal(a.query_start, a.subject_start)
                        - diagonal(b.query_start, b.subject_start))
                    .abs()
                        < 50
                        || (diagonal(a.query_end, a.subject_end)
                            - diagonal(b.query_end, b.subject_end))
                        .abs()
                            < 50
                })
        }) {
            continue;
        }
        kept.push(hsp);
    }
    *hits = kept;
}

/// Compatibility API returning every HSP for one query/subject.
pub fn search_subject(
    prepared: &PreparedQuery,
    subject: &SequenceRecord,
    backend: &dyn GappedAligner,
    match_run: MatchRun,
) -> Result<(Vec<Hsp>, SearchDiagnostics)> {
    let sequence = subject.sequence.unpack();
    let view = SequenceView {
        id: &subject.id,
        sequence: &sequence,
    };
    let (scored, mut diagnostics) = QueryBatch::new(std::slice::from_ref(prepared))
        .score_subject(view, backend, match_run)?
        .pop()
        .unwrap();
    let mut scratch = WorkerScratch::default();
    let hits = scored
        .iter()
        .map(|h| h.traceback(prepared, view, &mut scratch))
        .collect::<Result<Vec<_>>>()?;
    diagnostics.tracebacks += hits.len() as u64;
    Ok((hits, diagnostics))
}

pub fn sort_hsps(hsps: &mut [Hsp]) {
    hsps.sort_by(|a, b| {
        a.query_id
            .cmp(&b.query_id)
            .then(a.evalue.total_cmp(&b.evalue))
            .then(b.alignment.summary.score.cmp(&a.alignment.summary.score))
            .then(a.subject_id.cmp(&b.subject_id))
            .then(a.strand.cmp(&b.strand))
            .then(a.coordinates().cmp(&b.coordinates()))
    });
}

#[cfg(test)]
#[path = "search/reference.rs"]
mod reference;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        align::align_seeded,
        dna::{PackedDna, decode},
    };

    struct Scalar;
    impl GappedAligner for Scalar {
        fn name(&self) -> &'static str {
            "test-scalar"
        }
        fn score_batch(
            &self,
            requests: &[AlignRequest<'_>],
            options: AlignOptions,
        ) -> Result<Vec<AlignmentScore>> {
            requests
                .iter()
                .map(|&r| align_seeded(r, options, false).map(|a| a.summary))
                .collect()
        }
    }
    fn record(id: &str, ascii: &[u8]) -> SequenceRecord {
        SequenceRecord {
            id: id.into(),
            description: id.into(),
            sequence: PackedDna::from_ascii(ascii).unwrap(),
        }
    }

    // Independent old HashMap-based discovery order. QueryLookup is deliberately
    // unchanged by this optimization and remains a separate lookup oracle.
    fn old_seed_stream(
        batch: &QueryBatch<'_>,
        subject: SequenceView<'_>,
    ) -> (Vec<(usize, usize, usize)>, Vec<SearchDiagnostics>) {
        use std::collections::HashMap;
        let mut diagonals: Vec<HashMap<i64, (Option<usize>, usize)>> =
            batch.contexts.iter().map(|_| HashMap::new()).collect();
        let mut found = Vec::new();
        let mut diagnostics = vec![SearchDiagnostics::default(); batch.queries.len()];
        for group in &batch.groups {
            for (s, key) in Kmers::new(subject.sequence, None, group.word_size).unwrap() {
                for (stream, &(qi, ci)) in batch.contexts.iter().enumerate() {
                    let query = &batch.queries[qi];
                    if query.options.word_size != group.word_size {
                        continue;
                    }
                    let context = &query.contexts[ci];
                    for &q in crate::seed::SeedIndex::lookup(&context.lookup, key) {
                        let d = &mut diagnostics[qi];
                        d.seed_hits += 1;
                        let (last, covered) =
                            diagonals[stream].entry(s as i64 - q as i64).or_default();
                        if s < *covered {
                            continue;
                        }
                        if query.options.two_hit_window > 0 {
                            let previous = *last;
                            if previous.is_some_and(|p| s - p < group.word_size) {
                                continue;
                            }
                            *last = Some(s);
                            if previous.is_none_or(|p| s - p > query.options.two_hit_window) {
                                continue;
                            }
                        }
                        d.ungapped_extensions += 1;
                        let (score, range) = ungapped(
                            &context.sequence,
                            subject.sequence,
                            q,
                            s,
                            &query.options,
                            scalar_match_run,
                        );
                        *covered = range.end;
                        if score < query.options.min_ungapped_score {
                            continue;
                        }
                        d.gapped_candidates += 1;
                        found.push((stream, q, s));
                    }
                }
            }
        }
        (found, diagnostics)
    }

    #[test]
    fn seed_stream_matches_hash_oracle_for_every_word_size_and_reused_scratch() {
        let mut random = 4321u64;
        let mut text: Vec<_> = (0..137)
            .map(|_| {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                b"ACGT"[((random >> 32) % 4) as usize]
            })
            .collect();
        text[20..60].fill(b'T'); // Includes the all-ones 32-mer key.
        let mut mixed = text.clone();
        mixed[65..70].copy_from_slice(b"NRYSW");
        mixed[75..95].make_ascii_lowercase();
        let subject = record("s", &text.repeat(5));
        let sequence = subject.sequence.unpack();
        let view = SequenceView {
            id: &subject.id,
            sequence: &sequence,
        };
        let mut scratch = SeedScratch::default();
        for word_size in 1..=32 {
            for window in [0, 7, 64] {
                let queries: Vec<_> = [&text[..], &mixed[..], &text[..12]]
                    .into_iter()
                    .enumerate()
                    .map(|(i, bases)| {
                        PreparedQuery::new(
                            &record(&format!("q{i}"), bases),
                            SearchOptions {
                                word_size: if i == 2 { 33 - word_size } else { word_size },
                                strand: [
                                    StrandSelection::Both,
                                    StrandSelection::Minus,
                                    StrandSelection::Plus,
                                ][i],
                                dust: word_size % 3 == 0,
                                lowercase_masking: word_size % 2 == 0,
                                max_seed_occurrences: if word_size % 5 == 0 { 2 } else { 0 },
                                two_hit_window: window,
                                min_ungapped_score: if window == 7 { 40 } else { 0 },
                                ..Default::default()
                            },
                            sequence.len() as u64,
                            1,
                        )
                        .unwrap()
                    })
                    .collect();
                let batch = QueryBatch::new(&queries);
                scratch.configure(batch.contexts.iter().map(|&(qi, ci)| {
                    batch.queries[qi].contexts[ci]
                        .sequence
                        .len()
                        .saturating_sub(batch.queries[qi].options.word_size)
                        + 1
                }));
                let expected = old_seed_stream(&batch, view);
                for _ in 0..2 {
                    let mut actual = Vec::new();
                    let diagnostics = batch
                        .visit_candidates(view, scalar_match_run, &mut scratch, |stream, c| {
                            actual.push((stream, c.q, c.s));
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(
                        (&actual, &diagnostics),
                        (&expected.0, &expected.1),
                        "word={word_size}, window={window}"
                    );
                }
            }
        }
        // A long query uses sparse diagonal fallback; the short subject bounds work.
        let query = record("long", &text.repeat(500));
        let queries = [PreparedQuery::new(
            &query,
            SearchOptions {
                word_size: 32,
                dust: false,
                two_hit_window: 64,
                ..Default::default()
            },
            137,
            1,
        )
        .unwrap()];
        let batch = QueryBatch::new(&queries);
        scratch.configure(
            batch
                .contexts
                .iter()
                .map(|&(qi, ci)| batch.queries[qi].contexts[ci].sequence.len() - 31),
        );
        let small = SequenceView {
            id: "small",
            sequence: &sequence[..137],
        };
        let expected = old_seed_stream(&batch, small);
        let mut actual = Vec::new();
        let diagnostics = batch
            .visit_candidates(small, scalar_match_run, &mut scratch, |stream, c| {
                actual.push((stream, c.q, c.s));
                Ok(())
            })
            .unwrap();
        assert_eq!((actual, diagnostics), expected);
    }

    #[test]
    fn batched_lazy_search_matches_frozen_eager_search() {
        let mut random = 171u64;
        for case in 0..48 {
            let mut text: Vec<_> = (0..96)
                .map(|_| {
                    random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                    b"ACGT"[((random >> 32) % 4) as usize]
                })
                .collect();
            if case % 3 == 0 {
                text[45..55].fill(b'A');
            }
            if case % 4 == 0 {
                text[20..25].copy_from_slice(b"NRYSW");
            }
            let subject = record("subject", &text);
            let reverse: Vec<_> = reverse_complement(&subject.sequence.unpack())
                .into_iter()
                .map(decode)
                .collect();
            let mut insertion = text.clone();
            insertion.splice(60..60, b"GGGGGGGGGG".iter().copied());
            let lowercase = text.iter().map(u8::to_ascii_lowercase).collect::<Vec<_>>();
            let queries: Vec<_> = [&text, &reverse, &insertion, &lowercase]
                .iter()
                .enumerate()
                .map(|(i, seq)| {
                    PreparedQuery::new(
                        &record(&format!("q{i}"), seq),
                        SearchOptions {
                            word_size: [3, 7, 11, 32][(case + i) % 4],
                            strand: [
                                StrandSelection::Both,
                                StrandSelection::Plus,
                                StrandSelection::Minus,
                            ][(case + i) % 3],
                            dust: case % 2 == 0,
                            lowercase_masking: case % 3 == 0,
                            max_seed_occurrences: if case % 5 == 0 { 2 } else { 0 },
                            two_hit_window: if case % 2 == 0 { 40 } else { 0 },
                            min_ungapped_score: 0,
                            evalue: 1000.0,
                            align: AlignOptions {
                                band: [None, Some(1), Some(8), Some(64)][case % 4],
                                xdrop: [0, 7, 50][case % 3],
                                ..Default::default()
                            },
                            ..Default::default()
                        },
                        1000,
                        10,
                    )
                    .unwrap()
                })
                .collect();
            let sequence = subject.sequence.unpack();
            let view = SequenceView {
                id: &subject.id,
                sequence: &sequence,
            };
            let actual = QueryBatch::new(&queries)
                .score_subject(view, &Scalar, scalar_match_run)
                .unwrap();
            let mut scratch = WorkerScratch::default();
            for (q, (scored, diagnostics)) in queries.iter().zip(actual) {
                let (expected, mut reference_diagnostics) =
                    reference::search_subject(q, &subject, &Scalar, reference::scalar_match_run)
                        .unwrap();
                let hits = scored
                    .iter()
                    .map(|h| h.traceback(q, view, &mut scratch))
                    .collect::<Result<Vec<_>>>()
                    .unwrap();
                assert_eq!(hits, expected, "case={case} query={}", q.id);
                reference_diagnostics.tracebacks = 0;
                assert_eq!(diagnostics, reference_diagnostics);
                for (a, b) in hits.iter().zip(expected) {
                    assert_eq!(a.evalue.to_bits(), b.evalue.to_bits());
                    assert_eq!(a.bit_score.to_bits(), b.bit_score.to_bits());
                }
            }
        }
    }

    #[test]
    fn repeated_seeds_cross_candidate_batches_and_keep_budgets() {
        let query = record("q", &[b'A'; 40]);
        let subject = record("s", &[b'A'; 400]);
        for max_candidates in [1, 5, 100_000] {
            let prepared = PreparedQuery::new(
                &query,
                SearchOptions {
                    word_size: 3,
                    dust: false,
                    min_ungapped_score: 0,
                    evalue: 1000.0,
                    max_candidates,
                    ..Default::default()
                },
                400,
                1,
            )
            .unwrap();
            let expected = reference::search_subject(
                &prepared,
                &subject,
                &Scalar,
                reference::scalar_match_run,
            );
            let actual = search_subject(&prepared, &subject, &Scalar, scalar_match_run);
            match (expected, actual) {
                (Ok((a, da)), Ok((b, db))) => {
                    assert!(da.gapped_candidates > 256);
                    assert_eq!(a, b);
                    assert_eq!(da.gapped_candidates, db.gapped_candidates);
                    assert_eq!(db.tracebacks, b.len() as u64);
                }
                (Err(a), Err(b)) => assert_eq!(a.0, b.0),
                _ => panic!("different budget outcome"),
            }
        }
    }

    #[test]
    fn cross_subject_batches_match_eager_oracle_and_keep_pair_budgets() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        struct Counting {
            mixed: AtomicBool,
            largest: AtomicUsize,
        }
        impl GappedAligner for Counting {
            fn name(&self) -> &'static str {
                "counting"
            }
            fn score_batch(
                &self,
                requests: &[AlignRequest<'_>],
                options: AlignOptions,
            ) -> Result<Vec<AlignmentScore>> {
                self.largest.fetch_max(requests.len(), Ordering::Relaxed);
                if requests
                    .iter()
                    .any(|r| r.subject.as_ptr() != requests[0].subject.as_ptr())
                {
                    self.mixed.store(true, Ordering::Relaxed);
                }
                Scalar.score_batch(requests, options)
            }
        }
        let backend = Counting {
            mixed: AtomicBool::new(false),
            largest: AtomicUsize::new(0),
        };
        let records: Vec<_> = (0..9)
            .map(|i| record(&format!("s{i}"), &[b'A'; 35]))
            .collect();
        let sequences: Vec<_> = records.iter().map(|r| r.sequence.unpack()).collect();
        let views: Vec<_> = records
            .iter()
            .zip(&sequences)
            .map(|(r, s)| SequenceView {
                id: &r.id,
                sequence: s,
            })
            .collect();
        let queries: Vec<_> = [3, 7]
            .into_iter()
            .map(|k| {
                PreparedQuery::new(
                    &record(&format!("q{k}"), &[b'A'; 20]),
                    SearchOptions {
                        word_size: k,
                        dust: false,
                        min_ungapped_score: 0,
                        max_candidates: 64,
                        evalue: 1000.0,
                        ..Default::default()
                    },
                    315,
                    9,
                )
                .unwrap()
            })
            .collect();
        let results = QueryBatch::new(&queries)
            .score_subjects(&views, &backend, scalar_match_run)
            .unwrap();
        assert!(backend.mixed.load(Ordering::Relaxed));
        assert_eq!(backend.largest.load(Ordering::Relaxed), 256);
        let mut scratch = WorkerScratch::default();
        for (si, results) in results.into_iter().enumerate() {
            for (query, (scores, mut diagnostics)) in queries.iter().zip(results) {
                let (expected, mut d) = reference::search_subject(
                    query,
                    &records[si],
                    &Scalar,
                    reference::scalar_match_run,
                )
                .unwrap();
                let actual = scores
                    .iter()
                    .map(|h| h.traceback(query, views[si], &mut scratch))
                    .collect::<Result<Vec<_>>>()
                    .unwrap();
                assert_eq!(actual, expected);
                diagnostics.tracebacks = 0;
                d.tracebacks = 0;
                assert_eq!(diagnostics, d);
            }
        }
    }

    #[test]
    fn shared_query_queues_keep_options_seeds_order_and_per_pair_state() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
        struct Expected {
            query: usize,
            options: AlignOptions,
            word: usize,
            capacity: usize,
        }
        struct Shared {
            hint: usize,
            expected: std::collections::HashMap<usize, Expected>,
            mixed: AtomicBool,
            largest: AtomicUsize,
        }
        impl GappedAligner for Shared {
            fn name(&self) -> &'static str {
                "shared-test"
            }
            fn shared_query_batch_size(&self) -> Option<usize> {
                Some(self.hint)
            }
            fn score_batch(
                &self,
                requests: &[AlignRequest<'_>],
                options: AlignOptions,
            ) -> Result<Vec<AlignmentScore>> {
                let first = &self.expected[&(requests[0].query.as_ptr() as usize)];
                assert!(requests.len() <= self.hint.clamp(1, 1024).min(first.capacity));
                for r in requests {
                    let e = &self.expected[&(r.query.as_ptr() as usize)];
                    assert_eq!(options, e.options);
                    assert_eq!(r.word_size, e.word);
                    if e.query != first.query {
                        self.mixed.store(true, Relaxed);
                    }
                }
                self.largest.fetch_max(requests.len(), Relaxed);
                Scalar.score_batch(requests, options)
            }
        }
        let records: Vec<_> = (0..9)
            .map(|i| {
                record(
                    &format!("s{i}"),
                    &vec![if i % 2 == 0 { b'A' } else { b'T' }; 35 + i],
                )
            })
            .collect();
        let sequences: Vec<_> = records.iter().map(|r| r.sequence.unpack()).collect();
        let views: Vec<_> = records
            .iter()
            .zip(&sequences)
            .map(|(r, s)| SequenceView {
                id: &r.id,
                sequence: s,
            })
            .collect();
        let queries: Vec<_> = (0..6)
            .map(|qi| {
                let mut options = SearchOptions {
                    word_size: if qi % 2 == 0 { 3 } else { 7 },
                    strand: [
                        StrandSelection::Both,
                        StrandSelection::Plus,
                        StrandSelection::Minus,
                    ][qi % 3],
                    dust: false,
                    min_ungapped_score: 0,
                    min_score: qi as i32 * 2,
                    evalue: if qi == 2 { 0.001 } else { 1_000_000.0 },
                    max_candidates: 128,
                    ..Default::default()
                };
                if qi / 2 == 1 {
                    options.align.band = Some(1);
                }
                if qi / 2 == 2 {
                    options.align.scoring.gap_open = 4;
                }
                PreparedQuery::new(
                    &record(&format!("q{qi}"), &vec![b'A'; 20 + qi]),
                    options,
                    351,
                    9,
                )
                .unwrap()
            })
            .collect();
        let batch = QueryBatch::new(&queries);
        for hint in [0, 1, 7, 256, 1024, usize::MAX] {
            let expected = queries
                .iter()
                .enumerate()
                .flat_map(|(qi, q)| {
                    let capacity = 256
                        * queries
                            .iter()
                            .filter(|other| other.options.align == q.options.align)
                            .count();
                    q.contexts.iter().map(move |context| {
                        (
                            context.sequence.as_ptr() as usize,
                            Expected {
                                query: qi,
                                options: q.options.align,
                                word: q.options.word_size,
                                capacity,
                            },
                        )
                    })
                })
                .collect();
            let backend = Shared {
                hint,
                expected,
                mixed: AtomicBool::new(false),
                largest: AtomicUsize::new(0),
            };
            let mut seed_scratch = SeedScratch::default();
            for _ in 0..2 {
                let results = batch
                    .score_subjects_with_scratch(
                        &views,
                        &backend,
                        scalar_match_run,
                        &mut seed_scratch,
                    )
                    .unwrap();
                let mut trace_scratch = WorkerScratch::default();
                for (si, found) in results.into_iter().enumerate() {
                    for (query, (hits, mut d)) in queries.iter().zip(found) {
                        let (expected, mut reference_d) = reference::search_subject(
                            query,
                            &records[si],
                            &Scalar,
                            reference::scalar_match_run,
                        )
                        .unwrap();
                        let actual: Vec<_> = hits
                            .iter()
                            .map(|h| h.traceback(query, views[si], &mut trace_scratch).unwrap())
                            .collect();
                        assert_eq!(
                            actual, expected,
                            "hint={hint} query={} subject={si}",
                            query.id
                        );
                        d.tracebacks = 0;
                        reference_d.tracebacks = 0;
                        assert_eq!(d, reference_d);
                    }
                }
            }
            if hint >= 256 {
                assert!(backend.mixed.load(Relaxed));
                assert!(backend.largest.load(Relaxed) >= 256);
            }
        }
    }

    #[test]
    fn shared_queue_error_replays_original_flush_order_on_same_backend() {
        use std::sync::Mutex;
        struct Failing {
            shared: bool,
            seen: Mutex<Vec<usize>>,
        }
        impl GappedAligner for Failing {
            fn name(&self) -> &'static str {
                "failing"
            }
            fn shared_query_batch_size(&self) -> Option<usize> {
                self.shared.then_some(1)
            }
            fn score_batch(
                &self,
                requests: &[AlignRequest<'_>],
                _: AlignOptions,
            ) -> Result<Vec<AlignmentScore>> {
                let word = requests[0].word_size;
                self.seen.lock().unwrap().push(word);
                Err(Error(format!("failure at word {word}")))
            }
        }
        let subject = vec![0; 20];
        let views = [SequenceView {
            id: "s",
            sequence: &subject,
        }];
        for max_candidates in [1, 100] {
            let queries: Vec<_> = [7, 3]
                .into_iter()
                .map(|k| {
                    PreparedQuery::new(
                        &record(&format!("q{k}"), &[b'A'; 12]),
                        SearchOptions {
                            word_size: k,
                            strand: StrandSelection::Plus,
                            dust: false,
                            min_ungapped_score: 0,
                            max_candidates,
                            ..Default::default()
                        },
                        20,
                        1,
                    )
                    .unwrap()
                })
                .collect();
            let batch = QueryBatch::new(&queries);
            let old = Failing {
                shared: false,
                seen: Mutex::new(Vec::new()),
            };
            let shared = Failing {
                shared: true,
                seen: Mutex::new(Vec::new()),
            };
            let expected = batch
                .score_subjects(&views, &old, scalar_match_run)
                .unwrap_err();
            let actual = batch
                .score_subjects(&views, &shared, scalar_match_run)
                .unwrap_err();
            assert_eq!(actual.0, expected.0);
            assert_eq!(shared.seen.lock().unwrap()[0], 3);
            if max_candidates == 100 {
                assert_eq!(*shared.seen.lock().unwrap(), [3, 7]);
                assert_eq!(actual.0, "failure at word 7");
            } else {
                assert!(actual.0.contains("candidate budget exceeded"));
            }
        }
    }
}
