//! Frozen eager-traceback search, used only as a differential test oracle.
use super::{Hsp, PreparedQuery, QueryContext, SearchDiagnostics, SearchOptions};
use crate::{
    Error, Result,
    align::{AlignRequest, GappedAligner, align_seeded},
    dna::SequenceRecord,
    seed::{Kmers, SeedIndex},
};
use std::{collections::HashMap, ops::Range};

#[derive(Clone, Copy)]
struct Candidate {
    q: usize,
    s: usize,
}
#[derive(Default)]
struct DiagonalState {
    last_hit: Option<usize>,
    covered_until: usize,
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

pub fn search_subject(
    prepared: &PreparedQuery,
    subject: &SequenceRecord,
    backend: &dyn GappedAligner,
    match_run: MatchRun,
) -> Result<(Vec<Hsp>, SearchDiagnostics)> {
    let sequence = subject.sequence.unpack();
    let options = &prepared.options;
    let mut result = Vec::new();
    let mut diagnostics = SearchDiagnostics::default();
    for context in &prepared.contexts {
        let mut diagonals = HashMap::<i64, DiagonalState>::new();
        let mut batch = Vec::new();
        for (s, key) in Kmers::new(&sequence, None, options.word_size)? {
            for &q in context.lookup.lookup(key) {
                diagnostics.seed_hits += 1;
                let state = diagonals.entry(s as i64 - q as i64).or_default();
                if s < state.covered_until {
                    continue;
                }
                if options.two_hit_window > 0 {
                    let previous = state.last_hit;
                    // Overlapping words must not continuously reset the nonoverlap trigger.
                    if let Some(p) = previous
                        && s - p < options.word_size
                    {
                        continue;
                    }
                    state.last_hit = Some(s);
                    if previous.is_none_or(|p| s - p > options.two_hit_window) {
                        continue;
                    }
                }
                diagnostics.ungapped_extensions += 1;
                let (score, range) =
                    ungapped(&context.sequence, &sequence, q, s, options, match_run);
                state.covered_until = range.end;
                if score < options.min_ungapped_score {
                    continue;
                }
                diagnostics.gapped_candidates += 1;
                if diagnostics.gapped_candidates > options.max_candidates as u64 {
                    return Err(Error(format!(
                        "candidate budget exceeded for {} vs {}; increase --max-candidates or enable masking",
                        prepared.id, subject.id
                    )));
                }
                batch.push(Candidate { q, s });
                if batch.len() == 256 {
                    score_candidates(
                        prepared,
                        context,
                        subject,
                        &sequence,
                        &batch,
                        backend,
                        &mut result,
                        &mut diagnostics,
                    )?;
                    batch.clear();
                }
            }
        }
        score_candidates(
            prepared,
            context,
            subject,
            &sequence,
            &batch,
            backend,
            &mut result,
            &mut diagnostics,
        )?;
    }
    sort_hsps(&mut result);
    // Retain distinct HSPs, suppress exact duplicates and contained lower-scoring alignments.
    let mut kept: Vec<Hsp> = Vec::new();
    for hsp in result {
        let a = &hsp.alignment.summary;
        if kept.iter().any(|h| {
            let b = &h.alignment.summary;
            h.strand == hsp.strand
                && b.query_start <= a.query_start
                && b.query_end >= a.query_end
                && b.subject_start <= a.subject_start
                && b.subject_end >= a.subject_end
                && b.score >= a.score
        }) {
            continue;
        }
        kept.push(hsp);
    }
    Ok((kept, diagnostics))
}

#[allow(clippy::too_many_arguments)]
fn score_candidates(
    prepared: &PreparedQuery,
    context: &QueryContext,
    subject: &SequenceRecord,
    sequence: &[u8],
    batch: &[Candidate],
    backend: &dyn GappedAligner,
    result: &mut Vec<Hsp>,
    diagnostics: &mut SearchDiagnostics,
) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }
    let requests: Vec<_> = batch
        .iter()
        .map(|c| AlignRequest {
            query: &context.sequence,
            subject: sequence,
            query_anchor: c.q,
            subject_anchor: c.s,
            word_size: prepared.options.word_size,
        })
        .collect();
    let scores = backend.score_batch(&requests, prepared.options.align)?;
    if scores.len() != requests.len() {
        return Err(Error("backend returned wrong batch length".into()));
    }
    for (request, score) in requests.into_iter().zip(scores) {
        diagnostics.band_edge_candidates += u64::from(score.band_edge_reached);
        let evalue = prepared
            .statistics
            .evalue(score.score, prepared.search_space);
        if score.score < prepared.options.min_score || evalue > prepared.options.evalue {
            continue;
        }
        if result
            .iter()
            .any(|h| h.strand == context.strand && h.alignment.summary == score)
        {
            continue;
        }
        let alignment = align_seeded(request, prepared.options.align, true)?;
        if alignment.summary != score {
            return Err(Error(
                "backend score/endpoints differ from CPU traceback".into(),
            ));
        }
        diagnostics.tracebacks += 1;
        result.push(Hsp {
            query_id: prepared.id.clone(),
            subject_id: subject.id.clone(),
            query_len: prepared.len,
            subject_len: sequence.len(),
            strand: context.strand,
            bit_score: prepared.statistics.bits(score.score),
            evalue,
            alignment,
        });
    }
    Ok(())
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
