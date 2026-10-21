use blast_core::{
    Error, Result,
    align::{
        AlignOptions, AlignRequest, Alignment, AlignmentScore, GappedAligner, WorkerScratch,
        align_seeded_with_scratch,
    },
    search::{
        HitLimits, Hsp, MatchRun, PreparedQuery, QueryBatch, ScoredHsp, SearchDiagnostics,
        SeedScratch, scalar_match_run,
    },
};
use blast_db::SequenceStore;
use rayon::prelude::*;
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::Instant,
};

thread_local! {
    static SCORE_SCRATCH: RefCell<WorkerScratch> = RefCell::new(WorkerScratch::default());
}

#[cfg(target_arch = "aarch64")]
mod neon_dp;

/// Scalar DP oracle and portable fallback, with reusable thread-local rows.
pub struct ScalarAligner;
impl GappedAligner for ScalarAligner {
    fn name(&self) -> &'static str {
        "scalar"
    }
    fn score_batch(
        &self,
        requests: &[AlignRequest<'_>],
        options: AlignOptions,
    ) -> Result<Vec<AlignmentScore>> {
        SCORE_SCRATCH.with_borrow_mut(|scratch| {
            requests
                .iter()
                .map(|&r| align_seeded_with_scratch(r, options, false, scratch).map(|a| a.summary))
                .collect()
        })
    }
}

/// CPU dispatch uses SIMD only for sufficiently dense groups of independent alignments.
pub struct CpuAligner;
impl GappedAligner for CpuAligner {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn score_batch(
        &self,
        requests: &[AlignRequest<'_>],
        options: AlignOptions,
    ) -> Result<Vec<AlignmentScore>> {
        #[cfg(target_arch = "aarch64")]
        {
            neon_dp::score_batch(requests, options, false)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            ScalarAligner.score_batch(requests, options)
        }
    }
    fn traceback_batch(
        &self,
        requests: &[AlignRequest<'_>],
        options: AlignOptions,
        scratch: &mut WorkerScratch,
    ) -> Result<Vec<Alignment>> {
        #[cfg(target_arch = "aarch64")]
        {
            neon_dp::traceback_batch(requests, options, false, scratch)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            ScalarAligner.traceback_batch(requests, options, scratch)
        }
    }
}

/// Force the available SIMD kernel for full groups, retaining scalar tails and safety fallbacks.
pub struct SimdAligner;
impl GappedAligner for SimdAligner {
    fn name(&self) -> &'static str {
        if cfg!(target_arch = "aarch64") {
            "simd-neon"
        } else {
            "simd-scalar-fallback"
        }
    }
    fn score_batch(
        &self,
        requests: &[AlignRequest<'_>],
        options: AlignOptions,
    ) -> Result<Vec<AlignmentScore>> {
        #[cfg(target_arch = "aarch64")]
        {
            neon_dp::score_batch(requests, options, true)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            ScalarAligner.score_batch(requests, options)
        }
    }
    fn traceback_batch(
        &self,
        requests: &[AlignRequest<'_>],
        options: AlignOptions,
        scratch: &mut WorkerScratch,
    ) -> Result<Vec<Alignment>> {
        #[cfg(target_arch = "aarch64")]
        {
            neon_dp::traceback_batch(requests, options, true, scratch)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            ScalarAligner.traceback_batch(requests, options, scratch)
        }
    }
}

pub struct SearchEngine {
    pool: rayon::ThreadPool,
    match_run: MatchRun,
    subject_batch_size: usize,
    subject_batch_bases: usize,
}

pub type SearchResult = (Vec<Hsp>, SearchDiagnostics);

#[derive(Default, Debug)]
pub struct SearchTimings {
    /// Includes subject decoding, seed lookup and score-only DP.
    pub scoring_seconds: f64,
    pub selection_seconds: f64,
    /// Includes reloading only subjects selected for traceback.
    pub traceback_seconds: f64,
    pub scoring_subjects_decoded: usize,
    pub traceback_subjects_decoded: usize,
}

struct RankedHit {
    subject_index: usize,
    hit: ScoredHsp,
}

type ScoredResults = Vec<(Vec<RankedHit>, SearchDiagnostics)>;
type TraceJob = (usize, Vec<(usize, usize, ScoredHsp)>);
type TracedHit = (usize, usize, Hsp);

/// Decode each selected subject once and batch compatible seeds across subjects.
fn trace_block(
    jobs: &[TraceJob],
    queries: &[PreparedQuery],
    store: &dyn SequenceStore,
    backend: &dyn GappedAligner,
    scalar: &mut WorkerScratch,
    base_budget: usize,
) -> Result<Vec<TracedHit>> {
    let mut output = Vec::new();
    let mut start = 0;
    while start < jobs.len() {
        let mut subjects = Vec::new();
        let mut bases = 0;
        while start + subjects.len() < jobs.len() && bases < base_budget {
            let subject = store.decoded_record(jobs[start + subjects.len()].0)?;
            bases += subject.sequence.len();
            subjects.push(subject);
        }
        let mut requests = Vec::with_capacity(256);
        let mut metadata = Vec::with_capacity(256);
        let mut options = None;
        for (offset, subject) in subjects.iter().enumerate() {
            for (qi, rank, hit) in &jobs[start + offset].1 {
                let align = queries[*qi].options.align;
                if !requests.is_empty() && (options != Some(align) || requests.len() == 256) {
                    finish_trace_batch(
                        &requests,
                        &metadata,
                        queries,
                        backend,
                        scalar,
                        options.unwrap(),
                        &mut output,
                    )?;
                    requests.clear();
                    metadata.clear();
                }
                options = Some(align);
                requests.push(hit.traceback_request(&queries[*qi], subject.view())?);
                metadata.push((*qi, *rank, hit));
            }
        }
        if let Some(options) = options {
            finish_trace_batch(
                &requests,
                &metadata,
                queries,
                backend,
                scalar,
                options,
                &mut output,
            )?;
        }
        start += subjects.len();
    }
    Ok(output)
}

fn finish_trace_batch(
    requests: &[AlignRequest<'_>],
    metadata: &[(usize, usize, &ScoredHsp)],
    queries: &[PreparedQuery],
    backend: &dyn GappedAligner,
    scalar: &mut WorkerScratch,
    options: AlignOptions,
    output: &mut Vec<TracedHit>,
) -> Result<()> {
    let alignments = backend.traceback_batch(requests, options, scalar)?;
    if alignments.len() != requests.len() {
        return Err(Error(
            "traceback backend returned the wrong number of alignments".into(),
        ));
    }
    for ((qi, rank, hit), alignment) in metadata.iter().zip(alignments) {
        output.push((*qi, *rank, hit.finish_traceback(&queries[*qi], alignment)?));
    }
    Ok(())
}

fn select_ncbi_hits(hits: &mut Vec<RankedHit>, limits: HitLimits) {
    // BLAST emits contiguous subject groups, ranked by their best HSP, not a
    // global HSP ordering. Equal best scores prefer the larger database OID.
    let mut by_subject = BTreeMap::<usize, Vec<RankedHit>>::new();
    for hit in hits.drain(..) {
        by_subject.entry(hit.subject_index).or_default().push(hit);
    }
    let mut groups: Vec<_> = by_subject.into_values().collect();
    for group in &mut groups {
        group.sort_by(|a, b| a.hit.compare_ncbi(&b.hit));
        if limits.max_hsps != 0 {
            group.truncate(limits.max_hsps);
        }
    }
    groups.sort_by(|a, b| {
        a[0].hit
            .compare_ncbi_significance(&b[0].hit)
            .then(b[0].subject_index.cmp(&a[0].subject_index))
    });
    if limits.max_target_seqs != 0 {
        groups.truncate(limits.max_target_seqs);
    }
    hits.extend(groups.into_iter().flatten());
}

impl SearchEngine {
    pub fn new(threads: usize, simd: bool) -> Result<Self> {
        if threads == 0 {
            return Err(Error("thread count must be positive".into()));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| Error(e.to_string()))?;
        Ok(Self {
            pool,
            subject_batch_size: 128,
            subject_batch_bases: 1_048_576,
            match_run: if simd {
                simd_match_run
            } else {
                scalar_match_run
            },
        })
    }
    /// Soft base budget: whole records are never split, so one record may exceed it.
    pub fn with_subject_batching(mut self, records: usize, bases: usize) -> Result<Self> {
        if records == 0 || bases == 0 {
            return Err(Error("subject batch limits must be positive".into()));
        }
        self.subject_batch_size = records;
        self.subject_batch_bases = bases;
        Ok(self)
    }
    pub fn search(
        &self,
        query: &PreparedQuery,
        store: &dyn SequenceStore,
        backend: &dyn GappedAligner,
    ) -> Result<SearchResult> {
        Ok(self
            .search_batch(
                std::slice::from_ref(query),
                store,
                backend,
                HitLimits::default(),
            )?
            .pop()
            .unwrap())
    }

    pub fn search_batch(
        &self,
        queries: &[PreparedQuery],
        store: &dyn SequenceStore,
        backend: &dyn GappedAligner,
        limits: HitLimits,
    ) -> Result<Vec<SearchResult>> {
        Ok(self
            .search_batch_profiled(queries, store, backend, limits)?
            .0)
    }

    pub fn search_batch_profiled(
        &self,
        queries: &[PreparedQuery],
        store: &dyn SequenceStore,
        backend: &dyn GappedAligner,
        limits: HitLimits,
    ) -> Result<(Vec<SearchResult>, SearchTimings)> {
        if queries.is_empty() {
            return Ok((Vec::new(), SearchTimings::default()));
        }
        let start = Instant::now();
        let batch = QueryBatch::new(queries);
        let empty = || -> ScoredResults {
            queries
                .iter()
                .map(|_| (Vec::new(), SearchDiagnostics::default()))
                .collect()
        };
        // Independently scheduled bounded blocks enable cross-subject score batches.
        // No database-wide barrier between blocks; each subject is still decoded once.
        let mut scored = self.pool.install(|| {
            (0..store.len().div_ceil(self.subject_batch_size))
                .into_par_iter()
                .try_fold(
                    || (empty(), SeedScratch::default()),
                    |(mut results, mut seed_scratch), block| {
                        let mut index = block * self.subject_batch_size;
                        let end = index
                            .saturating_add(self.subject_batch_size)
                            .min(store.len());
                        while index < end {
                            let start = index;
                            let mut subjects = Vec::new();
                            let mut bases = 0;
                            while index < end && bases < self.subject_batch_bases {
                                let subject = store.decoded_record(index)?;
                                bases += subject.sequence.len();
                                subjects.push(subject);
                                index += 1;
                            }
                            let views: Vec<_> = subjects.iter().map(|s| s.view()).collect();
                            let found = batch.score_subjects_with_scratch(
                                &views,
                                backend,
                                self.match_run,
                                &mut seed_scratch,
                            )?;
                            for (offset, found) in found.into_iter().enumerate() {
                                for ((hits, diagnostics), (scored, d)) in
                                    results.iter_mut().zip(found)
                                {
                                    hits.extend(scored.into_iter().map(|hit| RankedHit {
                                        subject_index: start + offset,
                                        hit,
                                    }));
                                    diagnostics.merge(&d);
                                }
                            }
                        }
                        Ok::<_, Error>((results, seed_scratch))
                    },
                )
                .map(|result| result.map(|(results, _)| results))
                .try_reduce(empty, |mut left, right| {
                    for ((hits, diagnostics), (mut found, d)) in left.iter_mut().zip(right) {
                        hits.append(&mut found);
                        diagnostics.merge(&d);
                    }
                    Ok(left)
                })
        })?;
        let mut timings = SearchTimings {
            scoring_seconds: start.elapsed().as_secs_f64(),
            scoring_subjects_decoded: store.len(),
            ..Default::default()
        };
        let start = Instant::now();
        let mut jobs = BTreeMap::<usize, Vec<(usize, usize, ScoredHsp)>>::new();
        let mut outputs: Vec<Vec<Option<Hsp>>> = Vec::new();
        for (qi, (hits, diagnostics)) in scored.iter_mut().enumerate() {
            if queries[qi].options.ncbi_compatibility {
                select_ncbi_hits(hits, limits);
            } else {
                hits.sort_by(|a, b| {
                    a.hit
                        .compare(&b.hit)
                        .then(a.subject_index.cmp(&b.subject_index))
                });
                let mut counts = HashMap::<Arc<str>, usize>::new();
                hits.retain(|h| {
                    if !counts.contains_key(&h.hit.subject_id)
                        && limits.max_target_seqs != 0
                        && counts.len() >= limits.max_target_seqs
                    {
                        return false;
                    }
                    let n = counts.entry(Arc::clone(&h.hit.subject_id)).or_default();
                    *n += 1;
                    limits.max_hsps == 0 || *n <= limits.max_hsps
                });
            }
            outputs.push((0..hits.len()).map(|_| None).collect());
            diagnostics.tracebacks = hits.len() as u64;
            for (rank, h) in hits.drain(..).enumerate() {
                jobs.entry(h.subject_index)
                    .or_default()
                    .push((qi, rank, h.hit));
            }
        }
        timings.selection_seconds = start.elapsed().as_secs_f64();
        timings.traceback_subjects_decoded = jobs.len();
        let jobs: Vec<_> = jobs.into_iter().collect();
        let start = Instant::now();
        let traced = self.pool.install(|| {
            jobs.par_chunks(self.subject_batch_size.min(32))
                .map_init(WorkerScratch::default, |scratch, block| {
                    match trace_block(
                        block,
                        queries,
                        store,
                        backend,
                        scratch,
                        self.subject_batch_bases,
                    ) {
                        Ok(hits) => Ok(hits),
                        Err(error) => {
                            // Replay legacy subject/HSP order on failure. Preserve the first
                            // scalar/record error; never hide a new backend-only inconsistency.
                            for (index, hits) in block {
                                let subject = store.decoded_record(*index)?;
                                for (qi, _, h) in hits {
                                    h.traceback(&queries[*qi], subject.view(), scratch)?;
                                }
                            }
                            Err(error)
                        }
                    }
                })
                .collect::<Result<Vec<_>>>()
        })?;
        for (qi, rank, hsp) in traced.into_iter().flatten() {
            outputs[qi][rank] = Some(hsp);
        }
        timings.traceback_seconds = start.elapsed().as_secs_f64();
        let results = outputs
            .into_iter()
            .zip(scored)
            .map(|(hits, (_, diagnostics))| {
                (
                    hits.into_iter()
                        .map(|h| h.expect("every traceback job completed"))
                        .collect(),
                    diagnostics,
                )
            })
            .collect();
        Ok((results, timings))
    }
}

pub fn simd_match_run(query: &[u8], subject: &[u8]) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; helper loads only complete slices.
        unsafe { neon_match_run(query, subject) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime dispatch verifies AVX2, helper checks load bounds.
            return unsafe { avx2_match_run(query, subject) };
        }
        scalar_match_run(query, subject)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_match_run(query, subject)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_match_run(q: &[u8], s: &[u8]) -> usize {
    use std::arch::aarch64::*;
    let mut i = 0;
    while i + 16 <= q.len().min(s.len()) {
        // SAFETY: 16 bytes are in bounds in each input; unaligned loads are supported.
        let (a, b) = unsafe { (vld1q_u8(q.as_ptr().add(i)), vld1q_u8(s.as_ptr().add(i))) };
        let valid = vandq_u8(vceqq_u8(a, b), vcltq_u8(a, vdupq_n_u8(4)));
        if vminvq_u8(valid) != 255 {
            break;
        }
        i += 16;
    }
    i + scalar_match_run(&q[i..], &s[i..])
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_match_run(q: &[u8], s: &[u8]) -> usize {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 32 <= q.len().min(s.len()) {
        // SAFETY: 32 bytes are in bounds in each input; loadu accepts unaligned pointers.
        let (a, b) = unsafe {
            (
                _mm256_loadu_si256(q.as_ptr().add(i).cast()),
                _mm256_loadu_si256(s.as_ptr().add(i).cast()),
            )
        };
        let valid = _mm256_cmpeq_epi8(
            _mm256_and_si256(a, _mm256_set1_epi8(-4)),
            _mm256_setzero_si256(),
        );
        if _mm256_movemask_epi8(_mm256_and_si256(valid, _mm256_cmpeq_epi8(a, b))) != -1 {
            break;
        }
        i += 32;
    }
    i + scalar_match_run(&q[i..], &s[i..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use blast_core::{
        dna::{PackedDna, SequenceRecord},
        search::SearchOptions,
    };
    use blast_db::MemoryStore;

    #[test]
    fn trace_batches_keep_query_options_and_reject_backend_inconsistency() {
        use std::sync::Mutex;
        struct Probe {
            failure: usize,
            calls: Mutex<Vec<(AlignOptions, usize)>>,
        }
        impl GappedAligner for Probe {
            fn name(&self) -> &'static str {
                "trace-probe"
            }
            fn score_batch(
                &self,
                r: &[AlignRequest<'_>],
                o: AlignOptions,
            ) -> Result<Vec<AlignmentScore>> {
                ScalarAligner.score_batch(r, o)
            }
            fn traceback_batch(
                &self,
                r: &[AlignRequest<'_>],
                o: AlignOptions,
                s: &mut WorkerScratch,
            ) -> Result<Vec<Alignment>> {
                self.calls.lock().unwrap().push((o, r.len()));
                let mut output = CpuAligner.traceback_batch(r, o, s)?;
                match self.failure {
                    1 => {
                        output.pop();
                    }
                    2 => {
                        output[0].summary.score += 1;
                    }
                    3 => return Err(Error("injected traceback failure".into())),
                    _ => {}
                }
                Ok(output)
            }
        }
        let make = |id: String| {
            SequenceRecord {
            description: id.clone(), id,
            sequence: PackedDna::from_ascii(b"ACGTTGCAACCTGATCGTACGGATCCTAGACCTGGCATGAGTCCAGTAGCGATTCAGGCTAGCCTACGTAACGTCAGTCTAGGCTAGTCAA").unwrap(),
        }
        };
        let db = MemoryStore::new((0..35).map(|i| make(format!("s{i}"))).collect());
        let queries: Vec<_> = [7, 50, 7]
            .into_iter()
            .enumerate()
            .map(|(i, xdrop)| {
                PreparedQuery::new(
                    &make(format!("q{i}")),
                    SearchOptions {
                        dust: false,
                        align: AlignOptions {
                            xdrop,
                            band: None,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    db.total_bases(),
                    db.len(),
                )
                .unwrap()
            })
            .collect();
        let expected = SearchEngine::new(1, false)
            .unwrap()
            .with_subject_batching(1, 1)
            .unwrap()
            .search_batch(&queries, &db, &ScalarAligner, HitLimits::default())
            .unwrap();
        for threads in [1, 4] {
            let engine = SearchEngine::new(threads, true).unwrap();
            let probe = Probe {
                failure: 0,
                calls: Mutex::new(Vec::new()),
            };
            assert_eq!(
                engine
                    .search_batch(&queries, &db, &probe, HitLimits::default())
                    .unwrap(),
                expected
            );
            assert!(
                probe
                    .calls
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|(o, n)| [7, 50].contains(&o.xdrop) && *n <= 256)
            );
            probe.calls.lock().unwrap().clear();
            engine
                .search_batch(&queries[..1], &db, &probe, HitLimits::default())
                .unwrap();
            assert!(
                probe.calls.lock().unwrap().iter().any(|(_, n)| *n >= 4),
                "cross-subject SIMD batches required"
            );
            for (failure, message) in [(1, "wrong number"), (2, "score/endpoints"), (3, "injected")]
            {
                let probe = Probe {
                    failure,
                    calls: Mutex::new(Vec::new()),
                };
                let error = engine
                    .search_batch(&queries, &db, &probe, HitLimits::default())
                    .unwrap_err();
                assert!(error.0.contains(message), "{}", error.0);
            }
        }
    }

    #[test]
    fn simd_dispatch_preserves_tail_order_and_first_error() {
        let data: Vec<_> = (0..13).map(|i| vec![i as u8 % 4; 70 + i * 3]).collect();
        let requests: Vec<_> = data
            .iter()
            .enumerate()
            .map(|(i, q)| AlignRequest {
                query: q,
                subject: q,
                query_anchor: i,
                subject_anchor: i,
                word_size: 7,
            })
            .collect();
        for n in 0..=requests.len() {
            let expected = ScalarAligner
                .score_batch(&requests[..n], AlignOptions::default())
                .unwrap();
            assert_eq!(
                CpuAligner
                    .score_batch(&requests[..n], AlignOptions::default())
                    .unwrap(),
                expected
            );
            assert_eq!(
                SimdAligner
                    .score_batch(&requests[..n], AlignOptions::default())
                    .unwrap(),
                expected
            );
        }
        let mut invalid = requests[..4].to_vec();
        invalid[3].word_size = 0;
        let options = AlignOptions {
            max_cells: 1,
            ..Default::default()
        };
        let expected = ScalarAligner.score_batch(&invalid, options).unwrap_err().0;
        assert!(expected.contains("cell budget"));
        assert_eq!(
            SimdAligner.score_batch(&invalid, options).unwrap_err().0,
            expected
        );
        let options = AlignOptions {
            max_cells: 0,
            ..Default::default()
        };
        assert_eq!(
            SimdAligner.score_batch(&invalid, options).unwrap_err().0,
            ScalarAligner.score_batch(&invalid, options).unwrap_err().0
        );
        assert!(SimdAligner.score_batch(&[], options).unwrap().is_empty());
    }

    #[test]
    fn simd_search_is_identical_across_subject_block_sizes_and_threads() {
        let mut rng = 21u64;
        let query: Vec<_> = (0..96)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                b"ACGT"[(rng >> 32) as usize % 4]
            })
            .collect();
        let make = |id: String, sequence: &[u8]| SequenceRecord {
            description: id.clone(),
            id,
            sequence: PackedDna::from_ascii(sequence).unwrap(),
        };
        let records = (0..129)
            .map(|i| {
                let mut s = query.clone();
                if i % 3 == 0 {
                    s.splice(50..50, b"GGGG".iter().copied());
                }
                if i % 4 == 0 {
                    s[35] = b'N';
                }
                if i % 2 == 0 {
                    s = blast_core::dna::reverse_complement(
                        &PackedDna::from_ascii(&s).unwrap().unpack(),
                    )
                    .into_iter()
                    .map(blast_core::dna::decode)
                    .collect();
                }
                make(format!("s{:03}", 128 - i), &s)
            })
            .collect();
        let db = MemoryStore::new(records);
        let prepared: Vec<_> = [7, 11]
            .into_iter()
            .map(|k| {
                PreparedQuery::new(
                    &make(format!("q{k}"), &query),
                    SearchOptions {
                        word_size: k,
                        dust: false,
                        ..Default::default()
                    },
                    db.total_bases(),
                    db.len(),
                )
                .unwrap()
            })
            .collect();
        let limits = HitLimits {
            max_target_seqs: 5,
            max_hsps: 1,
        };
        let expected = SearchEngine::new(1, false)
            .unwrap()
            .with_subject_batching(1, 1)
            .unwrap()
            .search_batch(&prepared, &db, &ScalarAligner, limits)
            .unwrap();
        for threads in [1, 4] {
            for (records, bases) in [(3, 1), (3, 512), (128, 1_048_576)] {
                let engine = SearchEngine::new(threads, true)
                    .unwrap()
                    .with_subject_batching(records, bases)
                    .unwrap();
                for backend in [&CpuAligner as &dyn GappedAligner, &SimdAligner] {
                    assert_eq!(
                        engine
                            .search_batch(&prepared, &db, backend, limits)
                            .unwrap(),
                        expected
                    );
                }
            }
        }
        assert!(
            SearchEngine::new(1, true)
                .unwrap()
                .with_subject_batching(0, 10)
                .is_err()
        );
        assert!(
            SearchEngine::new(1, true)
                .unwrap()
                .with_subject_batching(10, 0)
                .is_err()
        );
    }

    #[test]
    fn lazy_limits_match_eager_selection_and_decode_once_per_batch() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Counted {
            inner: MemoryStore,
            reads: AtomicUsize,
        }
        impl SequenceStore for Counted {
            fn len(&self) -> usize {
                self.inner.len()
            }
            fn total_bases(&self) -> u64 {
                self.inner.total_bases()
            }
            fn record(&self, _: usize) -> Result<SequenceRecord> {
                panic!("search should use decoded views")
            }
            fn decoded_record(&self, index: usize) -> Result<blast_core::dna::DecodedSubject> {
                self.reads.fetch_add(1, Ordering::Relaxed);
                self.inner.decoded_record(index)
            }
        }
        let make = |id: String| SequenceRecord {
            description: id.clone(),
            id,
            sequence: PackedDna::from_ascii(b"ACGTTGCAACCTGATCGTACNNNNACGTTGCAACCTGATCGTAC")
                .unwrap(),
        };
        let db = Counted {
            inner: MemoryStore::new((0..30).rev().map(|i| make(format!("s{i:02}"))).collect()),
            reads: AtomicUsize::new(0),
        };
        let queries: Vec<_> = (0..3)
            .map(|i| {
                PreparedQuery::new(
                    &make(format!("q{i}")),
                    SearchOptions {
                        word_size: 7,
                        dust: false,
                        ..Default::default()
                    },
                    db.total_bases(),
                    db.len(),
                )
                .unwrap()
            })
            .collect();
        let full: Vec<_> = queries
            .iter()
            .map(|q| {
                SearchEngine::new(1, false)
                    .unwrap()
                    .search(q, &db.inner, &CpuAligner)
                    .unwrap()
                    .0
            })
            .collect();
        for threads in [1, 4] {
            let engine = SearchEngine::new(threads, true).unwrap();
            for limits in [
                HitLimits::default(),
                HitLimits {
                    max_target_seqs: 2,
                    max_hsps: 1,
                },
                HitLimits {
                    max_target_seqs: 3,
                    max_hsps: 0,
                },
            ] {
                db.reads.store(0, Ordering::Relaxed);
                let (actual, timing) = engine
                    .search_batch_profiled(&queries, &db, &CpuAligner, limits)
                    .unwrap();
                assert_eq!(timing.scoring_subjects_decoded, db.len());
                assert_eq!(
                    db.reads.load(Ordering::Relaxed),
                    db.len() + timing.traceback_subjects_decoded
                );
                for ((hits, diagnostics), original) in actual.iter().zip(&full) {
                    let mut expected = original.clone();
                    let mut subjects = std::collections::HashSet::new();
                    let mut counts = HashMap::<String, usize>::new();
                    expected.retain(|h| {
                        if limits.max_target_seqs != 0
                            && !subjects.contains(&h.subject_id)
                            && subjects.len() >= limits.max_target_seqs
                        {
                            return false;
                        }
                        subjects.insert(h.subject_id.clone());
                        let n = counts.entry(h.subject_id.clone()).or_default();
                        *n += 1;
                        limits.max_hsps == 0 || *n <= limits.max_hsps
                    });
                    assert_eq!(hits, &expected);
                    assert_eq!(diagnostics.tracebacks, hits.len() as u64);
                    if limits.max_target_seqs == 2 {
                        assert_eq!(
                            hits.iter()
                                .map(|h| h.subject_id.as_str())
                                .collect::<Vec<_>>(),
                            ["s00", "s01"]
                        );
                        assert!(diagnostics.tracebacks < original.len() as u64);
                    }
                }
            }
        }
    }
    #[test]
    fn simd_matches_scalar_at_every_boundary() {
        for n in 0..130 {
            let q = vec![2; n];
            for p in 0..=n {
                let mut s = q.clone();
                if p < n {
                    s[p] = 1;
                }
                assert_eq!(simd_match_run(&q, &s), scalar_match_run(&q, &s));
            }
            let q = vec![255; n];
            assert_eq!(simd_match_run(&q, &q), 0);
        }
    }
    #[test]
    fn deterministic_parallel_and_reverse_hits() {
        let query = SequenceRecord {
            id: "q".into(),
            description: "q".into(),
            sequence: PackedDna::from_ascii(b"ACGTTGCAACCTGATCGTAC").unwrap(),
        };
        let rc = blast_core::dna::reverse_complement(&query.sequence.unpack())
            .into_iter()
            .map(blast_core::dna::decode)
            .collect::<Vec<_>>();
        let subject = SequenceRecord {
            id: "s".into(),
            description: "s".into(),
            sequence: PackedDna::from_ascii(&rc).unwrap(),
        };
        let db = MemoryStore::new(vec![subject]);
        let options = SearchOptions {
            word_size: 7,
            dust: false,
            ..Default::default()
        };
        let q = PreparedQuery::new(&query, options, db.total_bases(), db.len()).unwrap();
        let a = SearchEngine::new(1, false)
            .unwrap()
            .search(&q, &db, &CpuAligner)
            .unwrap();
        let b = SearchEngine::new(4, true)
            .unwrap()
            .search(&q, &db, &CpuAligner)
            .unwrap();
        assert_eq!(a.0.len(), b.0.len());
        assert!(!a.0.is_empty());
        assert_eq!(a.0[0].coordinates(), (1, 20, 20, 1));
        assert_eq!(a.0[0].alignment.summary.score, 40);
        for (a, b) in a.0.iter().zip(&b.0) {
            assert_eq!(a.alignment, b.alignment);
        }
    }
}
