//! Four independent row-major, seed-anchored affine DPs in i32 NEON lanes.
//! No cell reordering within a lane; running-best X-drop and first-maximum ties match scalar.
use super::{SCORE_SCRATCH, ScalarAligner};
use blast_core::{
    Error, Result,
    align::{
        AlignOptions, AlignRequest, Alignment, AlignmentScore, GappedAligner, WorkerScratch,
        align_seeded_with_scratch,
    },
};
use std::{arch::aarch64::*, cell::RefCell};

const NEG: i32 = -1_000_000_000;
// Bound extra SIMD scratch. Larger windows use the scalar implementation without truncation.
const MAX_COLUMNS: usize = 16_384;
// Amortize transposition without preparing a whole, mostly unreachable subject flank.
const SUBJECT_BLOCK: usize = 32;
// Short flanks are cheaper to transpose once, with no preparation logic in the row loop.
const EAGER_COLUMNS: usize = 1024;
// A group exceeding these bounds is replayed by scalar traceback, never truncated.
// Each cell holds one byte per lane; rows hold only an interval start and arena offset.
const MAX_TRACE_CELLS: usize = 1 << 20;
const MAX_TRACE_ROWS: usize = 1 << 16;

struct TraceRow {
    start: usize,
    offset: usize,
}
#[derive(Default)]
struct TraceScratch {
    rows: Vec<TraceRow>,
    cells: Vec<[u8; 4]>,
}

#[derive(Default)]
struct Scratch {
    prev: Vec<int32x4_t>,
    curr: Vec<int32x4_t>,
    prev_gap: Vec<int32x4_t>,
    curr_gap: Vec<int32x4_t>,
    subject: Vec<int32x4_t>,
    trace: TraceScratch,
    #[cfg(test)]
    evaluated_columns: usize,
}
#[derive(Default)]
struct BatchScratch {
    kernel: Scratch,
    order: Vec<usize>,
    scores: Vec<Option<AlignmentScore>>,
    alignments: Vec<Option<Alignment>>,
}
// No borrowed sequences survive a call; only indices, owned scores and DP rows are retained.
thread_local! { static SCRATCH: RefCell<BatchScratch> = RefCell::new(BatchScratch::default()); }

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Side {
    score: i32,
    q: usize,
    s: usize,
    edge: bool,
}

fn dense(requests: &[AlignRequest<'_>]) -> bool {
    let left = requests.iter().map(|r| r.query_anchor).max().unwrap_or(0);
    let right = requests
        .iter()
        .map(|r| r.query.len() - r.query_anchor - r.word_size)
        .max()
        .unwrap_or(0);
    let actual: usize = requests.iter().map(|r| r.query.len() - r.word_size).sum();
    // Grouping policy affects performance only; sparse groups are computed exactly by scalar DP.
    requests.iter().all(|r| r.query.len() >= 64) && actual * 4 >= 3 * 4 * (left + right)
}

pub(super) fn score_batch(
    requests: &[AlignRequest<'_>],
    options: AlignOptions,
    force: bool,
) -> Result<Vec<AlignmentScore>> {
    if requests.len() < 4
        || options.validate().is_err()
        || requests.iter().any(|r| r.validate().is_err())
    {
        return ScalarAligner.score_batch(requests, options);
    }
    let result: Result<Vec<AlignmentScore>> = SCRATCH.with_borrow_mut(|scratch| {
        let BatchScratch {
            kernel,
            order,
            scores,
            ..
        } = scratch;
        // Similar flank shapes reduce idle SIMD lanes. Keep the exact previous grouping key.
        order.clear();
        order.extend(0..requests.len());
        order.sort_unstable_by_key(|&i| {
            let r = requests[i];
            (
                r.query.len() / 32,
                r.query_anchor / 32,
                r.subject_anchor.min(r.query_anchor) / 32,
                i,
            )
        });
        scores.clear();
        scores.resize_with(requests.len(), || None);
        for indices in order.chunks(4) {
            // Stack-only request gathering, including a padded tail which is never run as SIMD.
            let group = std::array::from_fn(|lane| requests[indices[lane.min(indices.len() - 1)]]);
            if indices.len() == 4 && (force || dense(&group)) {
                // SAFETY: NEON is available on the supported aarch64 hosts. Requests/options
                // are validated above. The kernel loads/stores exactly four array elements.
                if let Some(output) = unsafe { score_four(&group, options, kernel)? } {
                    for (&i, score) in indices.iter().zip(output) {
                        scores[i] = Some(score);
                    }
                    continue;
                }
            }
            // Sparse groups, oversized windows and tails reuse scalar scratch too, without
            // allocating a separate Vec for every fallback group.
            SCORE_SCRATCH.with_borrow_mut(|scalar| -> Result<()> {
                for &i in indices {
                    scores[i] = Some(
                        align_seeded_with_scratch(requests[i], options, false, scalar)?.summary,
                    );
                }
                Ok(())
            })?;
        }
        // The sole per-batch output allocation is required by the backend's owned Vec API.
        Ok(scores
            .drain(..)
            .map(|s| s.expect("every SIMD job completed"))
            .collect())
    });
    match result {
        Ok(scores) => Ok(scores),
        // Replay the entire original order, including scalar fallbacks, to preserve first errors.
        Err(_) => ScalarAligner.score_batch(requests, options),
    }
}

pub(super) fn traceback_batch(
    requests: &[AlignRequest<'_>],
    options: AlignOptions,
    force: bool,
    scalar: &mut WorkerScratch,
) -> Result<Vec<Alignment>> {
    if requests.len() < 4
        || options.validate().is_err()
        || requests.iter().any(|r| r.validate().is_err())
    {
        return ScalarAligner.traceback_batch(requests, options, scalar);
    }
    let result: Result<Vec<Alignment>> = SCRATCH.with_borrow_mut(|scratch| {
        let BatchScratch {
            kernel,
            order,
            alignments,
            ..
        } = scratch;
        order.clear();
        order.extend(0..requests.len());
        order.sort_unstable_by_key(|&i| {
            let r = requests[i];
            (
                r.query.len() / 32,
                r.query_anchor / 32,
                r.subject_anchor.min(r.query_anchor) / 32,
                i,
            )
        });
        alignments.clear();
        alignments.resize_with(requests.len(), || None);
        for indices in order.chunks(4) {
            let group = std::array::from_fn(|lane| requests[indices[lane.min(indices.len() - 1)]]);
            if indices.len() == 4 && (force || dense(&group)) {
                // SAFETY: four validated seeds/options on aarch64 NEON.
                if let Some(output) = unsafe { trace_four(&group, options, kernel)? } {
                    for (&i, alignment) in indices.iter().zip(output) {
                        alignments[i] = Some(alignment);
                    }
                    continue;
                }
            }
            for &i in indices {
                alignments[i] = Some(align_seeded_with_scratch(
                    requests[i],
                    options,
                    true,
                    scalar,
                )?);
            }
        }
        Ok(alignments
            .drain(..)
            .map(|a| a.expect("every traceback job completed"))
            .collect())
    });
    // Sorting must not change the first failure. Discard partial trace state and replay.
    result.or_else(|_| {
        SCRATCH.with_borrow_mut(|scratch| scratch.alignments.clear());
        ScalarAligner.traceback_batch(requests, options, scalar)
    })
}

#[target_feature(enable = "neon")]
unsafe fn trace_four(
    requests: &[AlignRequest<'_>; 4],
    options: AlignOptions,
    scratch: &mut Scratch,
) -> Result<Option<[Alignment; 4]>> {
    let left: [(&[u8], &[u8]); 4] = std::array::from_fn(|lane| {
        let r = requests[lane];
        let len = options.band.map_or(r.subject_anchor, |b| {
            r.subject_anchor.min(r.query_anchor.saturating_add(b))
        });
        (
            &r.query[..r.query_anchor],
            &r.subject[r.subject_anchor - len..r.subject_anchor],
        )
    });
    let right = std::array::from_fn(|lane| {
        let r = requests[lane];
        (
            &r.query[r.query_anchor + r.word_size..],
            &r.subject[r.subject_anchor + r.word_size..],
        )
    });
    // SAFETY: validated exact seeds/options and NEON, as for score_four.
    let Some(a) = (unsafe { extend_mode::<true, true>(&left, options, scratch)? }) else {
        return Ok(None);
    };
    let mut operations =
        std::array::from_fn(|i| Vec::with_capacity(a[i].q + a[i].s + requests[i].word_size));
    recover::<true>(&left, &a, &scratch.trace, &mut operations)?;
    for i in 0..4 {
        operations[i].extend(std::iter::repeat_n(b'=', requests[i].word_size));
    }
    // Recover the left side before reusing its arena for the right side.
    // SAFETY: same validated seeds/options and NEON.
    let Some(b) = (unsafe { extend_mode::<false, true>(&right, options, scratch)? }) else {
        return Ok(None);
    };
    recover::<false>(&right, &b, &scratch.trace, &mut operations)?;
    Ok(Some(std::array::from_fn(|i| {
        let r = requests[i];
        Alignment {
            summary: AlignmentScore {
                score: a[i].score + b[i].score + r.word_size as i32 * options.scoring.reward,
                query_start: r.query_anchor - a[i].q,
                query_end: r.query_anchor + r.word_size + b[i].q,
                subject_start: r.subject_anchor - a[i].s,
                subject_end: r.subject_anchor + r.word_size + b[i].s,
                band_edge_reached: a[i].edge || b[i].edge,
            },
            operations: std::mem::take(&mut operations[i]),
        }
    })))
}

fn recover<const REVERSE: bool>(
    flanks: &[(&[u8], &[u8]); 4],
    sides: &[Side; 4],
    trace: &TraceScratch,
    operations: &mut [Vec<u8>; 4],
) -> Result<()> {
    for lane in 0..4 {
        let (q, s) = flanks[lane];
        let ops = &mut operations[lane];
        let start = ops.len();
        let (mut i, mut j, mut state) = (sides[lane].q, sides[lane].s, 0);
        while i != 0 || j != 0 {
            if i == 0 {
                ops.push(b'D');
                j -= 1;
                continue;
            }
            if j == 0 {
                ops.push(b'I');
                i -= 1;
                continue;
            }
            let row = &trace.rows[i];
            let code = trace.cells[row.offset + j - row.start][lane];
            if state == 0 {
                state = code & 3;
            }
            match state {
                1 => {
                    ops.push(
                        if base::<REVERSE>(q, i - 1) < 4
                            && base::<REVERSE>(q, i - 1) == base::<REVERSE>(s, j - 1)
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
                    ops.push(b'I');
                    i -= 1;
                    state = if code & 4 != 0 { 2 } else { 0 };
                }
                3 => {
                    ops.push(b'D');
                    j -= 1;
                    state = if code & 8 != 0 { 3 } else { 0 };
                }
                _ => return Err(Error("invalid traceback state".into())),
            }
        }
        // Left extension already walks in forward original-query order; right must reverse.
        if !REVERSE {
            ops[start..].reverse();
        }
    }
    Ok(())
}

#[target_feature(enable = "neon")]
unsafe fn score_four(
    requests: &[AlignRequest<'_>; 4],
    options: AlignOptions,
    scratch: &mut Scratch,
) -> Result<Option<[AlignmentScore; 4]>> {
    let left: [(&[u8], &[u8]); 4] = std::array::from_fn(|lane| {
        let r = requests[lane];
        let len = options.band.map_or(r.subject_anchor, |b| {
            r.subject_anchor.min(r.query_anchor.saturating_add(b))
        });
        (
            &r.query[..r.query_anchor],
            &r.subject[r.subject_anchor - len..r.subject_anchor],
        )
    });
    let right: [(&[u8], &[u8]); 4] = std::array::from_fn(|lane| {
        let r = requests[lane];
        (
            &r.query[r.query_anchor + r.word_size..],
            &r.subject[r.subject_anchor + r.word_size..],
        )
    });
    // SAFETY: same validated inputs and NEON feature as this function.
    let Some(a) = (unsafe { extend::<true>(&left, options, scratch)? }) else {
        return Ok(None);
    };
    let Some(b) = (unsafe { extend::<false>(&right, options, scratch)? }) else {
        return Ok(None);
    };
    Ok(Some(std::array::from_fn(|i| {
        let r = requests[i];
        AlignmentScore {
            score: a[i].score + b[i].score + r.word_size as i32 * options.scoring.reward,
            query_start: r.query_anchor - a[i].q,
            query_end: r.query_anchor + r.word_size + b[i].q,
            subject_start: r.subject_anchor - a[i].s,
            subject_end: r.subject_anchor + r.word_size + b[i].s,
            band_edge_reached: a[i].edge || b[i].edge,
        }
    })))
}

#[inline]
fn base<const REVERSE: bool>(slice: &[u8], i: usize) -> i32 {
    slice[if REVERSE { slice.len() - 1 - i } else { i }] as i32
}

#[target_feature(enable = "neon")]
unsafe fn pack_subject<const REVERSE: bool>(
    flanks: &[(&[u8], &[u8]); 4],
    lengths: &[usize; 4],
    subject: &mut Vec<int32x4_t>,
    end: usize,
) {
    for j in subject.len()..end {
        let codes: [i32; 4] = std::array::from_fn(|lane| {
            if j < lengths[lane] {
                base::<REVERSE>(flanks[lane].1, j)
            } else {
                15
            }
        });
        // SAFETY: codes contains exactly four initialized i32 elements.
        subject.push(unsafe { vld1q_s32(codes.as_ptr()) });
    }
}

#[target_feature(enable = "neon")]
unsafe fn extend<const REVERSE: bool>(
    flanks: &[(&[u8], &[u8]); 4],
    options: AlignOptions,
    scratch: &mut Scratch,
) -> Result<Option<[Side; 4]>> {
    // SAFETY: preserve the caller's validated flanks and NEON feature.
    unsafe { extend_mode::<REVERSE, false>(flanks, options, scratch) }
}

#[target_feature(enable = "neon")]
unsafe fn extend_mode<const REVERSE: bool, const TRACE: bool>(
    flanks: &[(&[u8], &[u8]); 4],
    options: AlignOptions,
    scratch: &mut Scratch,
) -> Result<Option<[Side; 4]>> {
    let lazy = flanks.iter().any(|(q, s)| {
        options
            .band
            .map_or(s.len(), |b| s.len().min(q.len().saturating_add(b)))
            > EAGER_COLUMNS
    });
    // SAFETY: this dispatch preserves the caller's validated flanks and NEON feature.
    unsafe {
        if lazy {
            extend_rows::<REVERSE, true, TRACE>(flanks, options, scratch)
        } else {
            extend_rows::<REVERSE, false, TRACE>(flanks, options, scratch)
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn extend_rows<const REVERSE: bool, const LAZY: bool, const TRACE: bool>(
    flanks: &[(&[u8], &[u8]); 4],
    options: AlignOptions,
    scratch: &mut Scratch,
) -> Result<Option<[Side; 4]>> {
    if TRACE {
        scratch.trace.rows.clear();
        scratch.trace.cells.clear();
        scratch.trace.rows.push(TraceRow {
            start: 0,
            offset: 0,
        });
    }
    let query_len: [usize; 4] = std::array::from_fn(|i| flanks[i].0.len());
    let subject_len: [usize; 4] = std::array::from_fn(|i| {
        options.band.map_or(flanks[i].1.len(), |b| {
            flanks[i].1.len().min(query_len[i].saturating_add(b))
        })
    });
    let rows = *query_len.iter().max().unwrap();
    let columns = *subject_len.iter().max().unwrap();
    if rows == 0 {
        return Ok(Some([Side::default(); 4]));
    }
    if columns > MAX_COLUMNS {
        return Ok(None);
    }
    let band = options.band.unwrap_or(rows.max(columns));
    let score = options.scoring;
    let Scratch {
        prev,
        curr,
        prev_gap,
        curr_gap,
        subject,
        trace,
        #[cfg(test)]
        evaluated_columns,
    } = scratch;
    #[cfg(test)]
    {
        *evaluated_columns = 0;
    }
    let neg = vdupq_n_s32(NEG);
    let zero = vdupq_n_s32(0);
    let open = vdupq_n_s32(score.gap_open + score.gap_extend);
    let gap_extend = vdupq_n_s32(score.gap_extend);
    let drop = vdupq_n_s32(options.xdrop);
    let reward = vdupq_n_s32(score.reward);
    let penalty = vdupq_n_s32(score.penalty);
    for row in [&mut *prev, &mut *curr, &mut *prev_gap, &mut *curr_gap] {
        row.resize(columns + 1, neg);
    }
    subject.clear();
    if !LAZY {
        // SAFETY: same validated flanks and NEON feature as this function.
        unsafe { pack_subject::<REVERSE>(flanks, &subject_len, subject, columns) };
    }
    let mut prev_start = 0;
    let leading = options.xdrop.saturating_sub(score.gap_open).max(0) / score.gap_extend;
    let horizontal_reach = options.xdrop as usize / score.gap_extend as usize;
    let mut prev_end = columns.min(band).min(leading as usize);
    for j in 0..=prev_end {
        let h = -score.gap_open - score.gap_extend * j as i32;
        let values: [i32; 4] = std::array::from_fn(|lane| {
            if j == 0 {
                0
            } else if j <= subject_len[lane] && h >= -options.xdrop {
                h
            } else {
                NEG
            }
        });
        // SAFETY: values contains exactly four initialized i32 elements.
        prev[j] = unsafe { vld1q_s32(values.as_ptr()) };
        prev_gap[j] = neg;
    }
    let lengths: [i32; 4] = subject_len.map(|n| n as i32);
    let qlengths: [i32; 4] = query_len.map(|n| n as i32);
    // SAFETY: both arrays contain exactly four initialized i32 elements.
    let (n, m) = unsafe { (vld1q_s32(lengths.as_ptr()), vld1q_s32(qlengths.as_ptr())) };
    let mut active = vcgtq_s32(m, zero);
    let mut best_score = zero;
    let mut best_q = zero;
    let mut best_s = zero;
    let mut edge = vdupq_n_u32(0);
    let mut visited = [0usize; 4];
    // A full rectangle bounds every charged geometric row, even with no X-drop pruning.
    // If it fits, per-row lane extraction/accounting cannot change the budget outcome.
    let budget_prechecked = (0..4)
        .all(|lane| query_len[lane].saturating_mul(subject_len[lane] + 1) <= options.max_cells);
    for i in 1..=rows {
        let band_start = i.saturating_sub(band);
        let end = columns.min(i.saturating_add(band));
        if band_start > end {
            break;
        }
        let row = vdupq_n_s32(i as i32);
        active = vandq_u32(
            active,
            vandq_u32(
                vcleq_s32(row, m),
                vcgeq_s32(n, vdupq_n_s32(band_start as i32)),
            ),
        );
        if vmaxvq_u32(active) == 0 {
            break;
        }
        if !budget_prechecked {
            let mut active_lanes = [0u32; 4];
            // SAFETY: destination contains space for exactly four u32 elements.
            unsafe {
                vst1q_u32(active_lanes.as_mut_ptr(), active);
            }
            for lane in 0..4 {
                if active_lanes[lane] != 0 {
                    // Charge the same geometric row as scalar/reference, not only evaluated cells.
                    visited[lane] += subject_len[lane].min(i.saturating_add(band)) - band_start + 1;
                    if visited[lane] > options.max_cells {
                        return Err(Error(
                            "alignment cell budget exceeded; increase --max-cells or reduce --band"
                                .into(),
                        ));
                    }
                }
            }
        }
        // After prev_end + 1 there is no diagonal/vertical predecessor. Any horizontal
        // chain loses at least gap_extend per column and its starting score is <= best.
        // Thus at most xdrop/gap_extend further columns survive; include the first dead
        // column too. This bounds preparation only: evaluation and budget stay unchanged.
        let needed = end.min(prev_end + 2 + horizontal_reach);
        if LAZY && needed > subject.len() {
            let packed_end = columns.min(needed + SUBJECT_BLOCK - 1);
            // SAFETY: same validated flanks and NEON feature as this function.
            unsafe { pack_subject::<REVERSE>(flanks, &subject_len, subject, packed_end) };
        }
        let codes: [i32; 4] = std::array::from_fn(|lane| {
            if i <= query_len[lane] {
                base::<REVERSE>(flanks[lane].0, i - 1)
            } else {
                15
            }
        });
        // SAFETY: codes contains exactly four initialized i32 elements.
        let query = unsafe { vld1q_s32(codes.as_ptr()) };
        let valid_query = vcltq_s32(query, vdupq_n_s32(4));
        let mut horizontal = neg;
        let mut left = neg;
        let mut alive = vdupq_n_u32(0);
        // A conservative union of the four live intervals. No lane is truncated
        // to another lane's X-drop frontier; masks keep scores/best independent.
        let start = band_start.max(prev_start);
        if TRACE {
            // Same conservative reachable-column bound as lazy packing, not a truncation.
            // Check before allocating a row; the scalar fallback retains original budgets.
            if trace.rows.len() >= MAX_TRACE_ROWS
                || trace.cells.len() + needed.saturating_sub(start) + 1 > MAX_TRACE_CELLS
            {
                return Ok(None);
            }
            trace.rows.push(TraceRow {
                start,
                offset: trace.cells.len(),
            });
        }
        let mut first_live = usize::MAX;
        let mut last_live = 0;
        for j in start..=end {
            #[cfg(test)]
            {
                *evaluated_columns += 1;
            }
            let column = vdupq_n_s32(j as i32);
            let valid = vandq_u32(active, vcleq_s32(column, n));
            let (mut h, mut vertical);
            let mut code = vdupq_n_u32(2 | 4);
            if j == 0 {
                h = vdupq_n_s32(-score.gap_open - score.gap_extend * i as i32);
                vertical = h;
            } else {
                let (above, above_gap) = if j >= prev_start && j <= prev_end {
                    (prev[j], prev_gap[j])
                } else {
                    (neg, neg)
                };
                let e_open = vsubq_s32(left, open);
                let e_extend = vsubq_s32(horizontal, gap_extend);
                let f_open = vsubq_s32(above, open);
                let f_extend = vsubq_s32(above_gap, gap_extend);
                horizontal = vmaxq_s32(e_open, e_extend);
                vertical = vmaxq_s32(f_open, f_extend);
                let diag = if j > prev_start && j - 1 <= prev_end {
                    prev[j - 1]
                } else {
                    neg
                };
                let substitution = vbslq_s32(
                    vandq_u32(valid_query, vceqq_s32(query, subject[j - 1])),
                    reward,
                    penalty,
                );
                let diagonal = vaddq_s32(diag, substitution);
                if TRACE {
                    code = vbslq_u32(
                        vcgtq_s32(vertical, diagonal),
                        vdupq_n_u32(2),
                        vdupq_n_u32(1),
                    );
                    h = vmaxq_s32(diagonal, vertical);
                    code = vbslq_u32(vcgtq_s32(horizontal, h), vdupq_n_u32(3), code);
                    h = vmaxq_s32(h, horizontal);
                    code = vorrq_u32(code, vandq_u32(vcgtq_s32(f_extend, f_open), vdupq_n_u32(4)));
                    code = vorrq_u32(code, vandq_u32(vcgtq_s32(e_extend, e_open), vdupq_n_u32(8)));
                } else {
                    h = vmaxq_s32(diagonal, vmaxq_s32(vertical, horizontal));
                }
            }
            let live = vandq_u32(valid, vcleq_s32(vsubq_s32(best_score, h), drop));
            if TRACE {
                let narrow = vmovn_u16(vcombine_u16(
                    vmovn_u32(vandq_u32(code, live)),
                    vdup_n_u16(0),
                ));
                trace
                    .cells
                    .push(vget_lane_u32::<0>(vreinterpret_u32_u8(narrow)).to_le_bytes());
            }
            h = vbslq_s32(live, h, neg);
            vertical = vbslq_s32(live, vertical, neg);
            horizontal = vbslq_s32(live, horizontal, neg);
            alive = vorrq_u32(alive, live);
            if options.band.is_some() && i.abs_diff(j) == band {
                edge = vorrq_u32(
                    edge,
                    vandq_u32(live, vandq_u32(vcltq_s32(row, m), vcltq_s32(column, n))),
                );
            }
            // Strict comparison retains the first row-major endpoint at equal scores.
            let better = vandq_u32(live, vcgtq_s32(h, best_score));
            best_score = vbslq_s32(better, h, best_score);
            best_q = vbslq_s32(better, row, best_q);
            best_s = vbslq_s32(better, column, best_s);
            curr[j] = h;
            curr_gap[j] = vertical;
            left = h;
            if vmaxvq_u32(live) != 0 {
                first_live = first_live.min(j);
                last_live = j;
            } else if j > prev_end {
                // The previous frontier can feed prev_end + 1 diagonally. Once
                // that column and all horizontal chains die, later columns are dead.
                break;
            }
        }
        std::mem::swap(prev, curr);
        std::mem::swap(prev_gap, curr_gap);
        active = vandq_u32(active, alive);
        if first_live == usize::MAX {
            break;
        }
        prev_start = first_live;
        prev_end = last_live;
    }
    let (mut scores, mut q, mut s, mut edges) = ([0i32; 4], [0i32; 4], [0i32; 4], [0u32; 4]);
    // SAFETY: each destination contains space for exactly four elements of the stored type.
    unsafe {
        vst1q_s32(scores.as_mut_ptr(), best_score);
        vst1q_s32(q.as_mut_ptr(), best_q);
        vst1q_s32(s.as_mut_ptr(), best_s);
        vst1q_u32(edges.as_mut_ptr(), edge);
    }
    Ok(Some(std::array::from_fn(|i| Side {
        score: scores[i],
        q: q[i] as usize,
        s: s[i] as usize,
        edge: edges[i] != 0,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use blast_core::align::ScoringScheme;

    #[test]
    fn subject_packing_is_lazy_and_reset_for_each_direction_and_lane_shape() {
        let dirty = vec![3; 1024];
        let query = [0; 64];
        let mut scratch = Scratch::default();
        let options = AlignOptions {
            band: None,
            xdrop: 0,
            ..Default::default()
        };
        for reverse in [false, true, false] {
            // Leave a large, incompatible cache from a previous extension.
            let flanks = [(dirty.as_slice(), dirty.as_slice()); 4];
            // SAFETY: valid flanks/options on aarch64 NEON.
            unsafe { extend::<false>(&flanks, options, &mut scratch) }.unwrap();
            assert_eq!(scratch.subject.len(), dirty.len());

            let subjects: Vec<Vec<u8>> = [8192, 4096, 0, 17]
                .into_iter()
                .map(|len| {
                    let mut subject = vec![1; len];
                    let matched = len.min(query.len());
                    if reverse {
                        subject[len - matched..].fill(0);
                    } else {
                        subject[..matched].fill(0);
                    }
                    subject
                })
                .collect();
            let qlengths = [64, 33, 0, 60];
            let flanks =
                std::array::from_fn(|lane| (&query[..qlengths[lane]], subjects[lane].as_slice()));
            // SAFETY: valid flanks/options on aarch64 NEON, including empty/short lanes.
            let sides = unsafe {
                if reverse {
                    extend::<true>(&flanks, options, &mut scratch)
                } else {
                    extend::<false>(&flanks, options, &mut scratch)
                }
            }
            .unwrap()
            .unwrap();
            for lane in 0..4 {
                let matched = qlengths[lane].min(subjects[lane].len());
                assert_eq!(
                    sides[lane],
                    Side {
                        score: matched as i32 * 2,
                        q: matched,
                        s: matched,
                        edge: false,
                    }
                );
            }
            assert!(scratch.subject.len() <= query.len() + SUBJECT_BLOCK);
            assert!(scratch.subject.len() < subjects[0].len() / 64);
        }
    }

    #[test]
    fn reusable_batch_buffers_preserve_order_after_errors_and_shape_changes() {
        let data: Vec<_> = (0..37).map(|i| vec![i as u8 % 4; 16 + i * 5]).collect();
        let mut requests: Vec<_> = data
            .iter()
            .enumerate()
            .map(|(i, q)| AlignRequest {
                query: q,
                subject: q,
                query_anchor: i % 9,
                subject_anchor: i % 9,
                word_size: 7,
            })
            .collect();
        let options = AlignOptions::default();
        score_batch(&requests, options, true).unwrap();
        let capacity = || SCRATCH.with_borrow(|s| (s.order.capacity(), s.scores.capacity()));
        let initial = capacity();
        for force in [false, true] {
            requests.reverse();
            for len in [37, 4, 17, 1, 36, 0, 9] {
                let input = &requests[..len];
                assert_eq!(
                    score_batch(input, options, force).unwrap(),
                    ScalarAligner.score_batch(input, options).unwrap()
                );
                let limited = AlignOptions {
                    max_cells: 1,
                    ..options
                };
                assert_eq!(
                    score_batch(input, limited, force).map_err(|e| e.0),
                    ScalarAligner.score_batch(input, limited).map_err(|e| e.0)
                );
                assert_eq!(
                    score_batch(input, options, force).unwrap(),
                    ScalarAligner.score_batch(input, options).unwrap()
                );
                assert_eq!(
                    capacity(),
                    initial,
                    "batch buffers must retain their allocation"
                );
            }
        }
    }

    #[test]
    fn eager_and_lazy_kernels_match_scalar_across_budget_and_packing_boundaries() {
        let mut rng = 8191u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 32) as u8 % 4
        };
        let mut scratch = Scratch::default();
        for length in [31, 32, 33, EAGER_COLUMNS, EAGER_COLUMNS + 1, 8192] {
            let data: Vec<_> = (0..4)
                .map(|lane| {
                    let mut query: Vec<_> = (0..96 + lane * 9).map(|_| next()).collect();
                    let mut subject: Vec<_> = (0..length + query.len()).map(|_| next()).collect();
                    let qa = 40 + lane;
                    let sa = if lane % 2 == 0 { length.max(qa) } else { qa };
                    subject[sa - qa..sa - qa + query.len()].copy_from_slice(&query);
                    // Independent lanes, ambiguous bases, mismatches and a post-seed insertion.
                    query[qa - 17] = 4;
                    subject[sa - 23] = (subject[sa - 23] + 1) % 4;
                    subject.splice(sa + 12..sa + 12, std::iter::repeat_n(4, lane * 3));
                    (query, subject, qa, sa)
                })
                .collect();
            let requests = std::array::from_fn(|lane| AlignRequest {
                query: &data[lane].0,
                subject: &data[lane].1,
                query_anchor: data[lane].2,
                subject_anchor: data[lane].3,
                word_size: 7,
            });
            for band in [None, Some(0), Some(64), Some(EAGER_COLUMNS + 1)] {
                for xdrop in [0, 7, 50, 1000] {
                    for max_cells in [1, 40, 20_000_000] {
                        let options = AlignOptions {
                            band,
                            xdrop,
                            max_cells,
                            ..Default::default()
                        };
                        // SAFETY: valid exact seeds/options on aarch64 NEON.
                        let actual = unsafe { score_four(&requests, options, &mut scratch) }
                            .map(|s| s.expect("bounded test windows must use SIMD").to_vec())
                            .map_err(|e| e.0);
                        let expected = ScalarAligner
                            .score_batch(&requests, options)
                            .map_err(|e| e.0);
                        assert_eq!(actual, expected, "length={length} options={options:?}");
                        // SAFETY: same validated flanks, including both lazy directions.
                        let traced = unsafe { trace_four(&requests, options, &mut scratch) }
                            .map(|a| a.expect("bounded trace must not fallback").to_vec())
                            .map_err(|e| e.0);
                        assert_eq!(
                            traced,
                            ScalarAligner
                                .traceback_batch(&requests, options, &mut WorkerScratch::default())
                                .map_err(|e| e.0),
                            "trace length={length} {options:?}"
                        );
                    }
                }
            }
        }
        // Exhaust every small limit, including the full-rectangle fast-path boundary.
        let query = [0; 9];
        let subject = [0; 12];
        let requests = std::array::from_fn(|i| AlignRequest {
            query: &query[..query.len() - i],
            subject: &subject,
            query_anchor: i,
            subject_anchor: i,
            word_size: 1,
        });
        for band in [None, Some(0), Some(3)] {
            for xdrop in [0, 50] {
                for max_cells in 1..=query.len() * (subject.len() + 1) + 1 {
                    let options = AlignOptions {
                        band,
                        xdrop,
                        max_cells,
                        ..Default::default()
                    };
                    // SAFETY: valid exact seeds/options on aarch64 NEON.
                    let actual = unsafe { score_four(&requests, options, &mut scratch) }
                        .map(|s| s.unwrap().to_vec())
                        .map_err(|e| e.0);
                    assert_eq!(
                        actual,
                        ScalarAligner
                            .score_batch(&requests, options)
                            .map_err(|e| e.0),
                        "budget boundary: {options:?}"
                    );
                    // SAFETY: same validated seeds/options on aarch64 NEON.
                    let traced = unsafe { trace_four(&requests, options, &mut scratch) }
                        .map(|a| a.unwrap().to_vec())
                        .map_err(|e| e.0);
                    assert_eq!(
                        traced,
                        ScalarAligner
                            .traceback_batch(&requests, options, &mut WorkerScratch::default())
                            .map_err(|e| e.0),
                        "trace budget: {options:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn active_union_frontier_skips_dead_columns_and_preserves_divergent_lanes() {
        let mut random = 19u64;
        let query: Vec<u8> = (0..512)
            .map(|_| {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                (random >> 32) as u8 % 4
            })
            .collect();
        let mut scratch = Scratch::default();
        for xdrop in [1_000_000, 0, 50, 0] {
            let options = AlignOptions {
                xdrop,
                ..Default::default()
            };
            for reverse in [false, true] {
                let flanks = [(query.as_slice(), query.as_slice()); 4];
                // SAFETY: valid sequences/options, four lanes, and aarch64 NEON.
                let sides = unsafe {
                    if reverse {
                        extend::<true>(&flanks, options, &mut scratch)
                    } else {
                        extend::<false>(&flanks, options, &mut scratch)
                    }
                }
                .unwrap()
                .unwrap();
                assert!(
                    sides
                        .iter()
                        .all(|s| s.score == 1024 && s.q == 512 && s.s == 512)
                );
                if xdrop == 0 {
                    assert!(scratch.evaluated_columns <= 3 * query.len());
                }
            }
        }

        let subjects: Vec<Vec<u8>> = [0, 3, 17, 49]
            .into_iter()
            .map(|gap| {
                query[..32]
                    .iter()
                    .copied()
                    .chain(std::iter::repeat_n(4, gap))
                    .chain(query[32..].iter().copied())
                    .collect()
            })
            .collect();
        for band in [None, Some(0), Some(8), Some(64)] {
            for xdrop in [0, 7, 25, 100] {
                for reverse in [false, true] {
                    let requests = std::array::from_fn(|lane| AlignRequest {
                        query: &query[..query.len() - lane * 30],
                        subject: &subjects[lane],
                        query_anchor: if reverse { 24 } else { 0 },
                        subject_anchor: if reverse { 24 } else { 0 },
                        word_size: 7,
                    });
                    let options = AlignOptions {
                        band,
                        xdrop,
                        scoring: ScoringScheme {
                            gap_open: 0,
                            gap_extend: 1,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    // SAFETY: four valid exact seeds and options on aarch64 NEON.
                    let actual = unsafe { score_four(&requests, options, &mut scratch) }
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        actual.to_vec(),
                        ScalarAligner.score_batch(&requests, options).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn kernel_matches_scalar_for_unequal_lanes_xdrop_band_ties_and_budgets() {
        let mut rng = 771u64;
        let mut next = |n: usize| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 32) as usize % n
        };
        let mut scratch = Scratch::default();
        for case in 0..1200 {
            let k = 1 + next(32);
            let data: Vec<_> = (0..4)
                .map(|lane| {
                    let qlen = k + if case % 9 == lane { 0 } else { next(120) };
                    let slen = k + next(150);
                    let mut q: Vec<_> = (0..qlen)
                        .map(|_| next(if case % 5 == 0 { 15 } else { 4 }) as u8)
                        .collect();
                    let mut s: Vec<_> = (0..slen).map(|_| next(4) as u8).collect();
                    if lane != case % 4 {
                        s[..qlen.min(slen)].copy_from_slice(&q[..qlen.min(slen)]);
                    }
                    let qa = next(qlen - k + 1);
                    let sa = if lane % 2 == 0 && qa + k <= slen {
                        qa
                    } else {
                        next(slen - k + 1)
                    };
                    for j in 0..k {
                        q[qa + j] = next(4) as u8;
                        s[sa + j] = q[qa + j];
                    }
                    (q, s, qa, sa)
                })
                .collect();
            let requests = std::array::from_fn(|i| AlignRequest {
                query: &data[i].0,
                subject: &data[i].1,
                query_anchor: data[i].2,
                subject_anchor: data[i].3,
                word_size: k,
            });
            let options = AlignOptions {
                band: [None, Some(0), Some(1), Some(8), Some(64)][case % 5],
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
            // SAFETY: four valid seeds, validated limits/scoring, and aarch64 NEON.
            let actual = unsafe { score_four(&requests, options, &mut scratch) }
                .map(|scores| {
                    scores
                        .expect("small test inputs must execute SIMD, not fallback")
                        .to_vec()
                })
                .map_err(|e| e.0);
            let expected = ScalarAligner
                .score_batch(&requests, options)
                .map_err(|e| e.0);
            assert_eq!(actual, expected, "case={case} options={options:?}");
            // Compare complete operations, not only score/endpoints or a normalized CIGAR.
            // SAFETY: same validated four seeds/options; deliberately reuse dirty scratch.
            let traced = unsafe { trace_four(&requests, options, &mut scratch) }
                .map(|a| a.expect("small trace must not fallback").to_vec())
                .map_err(|e| e.0);
            let expected = ScalarAligner
                .traceback_batch(&requests, options, &mut WorkerScratch::default())
                .map_err(|e| e.0);
            assert_eq!(traced, expected, "trace case={case} options={options:?}");
        }
    }

    #[test]
    fn traceback_bounds_fall_back_without_truncation_and_recover_after_errors() {
        let mut kernel = Scratch::default();
        let mut scalar = WorkerScratch::default();
        for (qlen, slen) in [(1100, 1100), (MAX_TRACE_ROWS + 2, 1), (16, MAX_COLUMNS + 2)] {
            let query = vec![0; qlen];
            let subject = vec![0; slen];
            let requests = [AlignRequest {
                query: &query,
                subject: &subject,
                query_anchor: 0,
                subject_anchor: 0,
                word_size: 1,
            }; 4];
            let options = AlignOptions {
                band: None,
                xdrop: 1_000_000,
                ..Default::default()
            };
            // SAFETY: valid exact seeds/options and NEON. Every resource bound must fallback.
            assert!(
                unsafe { trace_four(&requests, options, &mut kernel) }
                    .unwrap()
                    .is_none()
            );
            assert_eq!(
                traceback_batch(&requests, options, true, &mut scalar).unwrap(),
                ScalarAligner
                    .traceback_batch(&requests, options, &mut scalar)
                    .unwrap()
            );
        }
        let data: Vec<_> = (0..13).map(|i| vec![i as u8 % 4; 70 + i * 9]).collect();
        let mut requests: Vec<_> = data
            .iter()
            .enumerate()
            .rev()
            .map(|(i, q)| AlignRequest {
                query: q,
                subject: q,
                query_anchor: i * 3,
                subject_anchor: i * 3,
                word_size: 7,
            })
            .collect();
        for force in [false, true] {
            for n in 0..=requests.len() {
                for max_cells in [1, 40, 20_000_000] {
                    let options = AlignOptions {
                        max_cells,
                        ..Default::default()
                    };
                    assert_eq!(
                        traceback_batch(&requests[..n], options, force, &mut scalar)
                            .map_err(|e| e.0),
                        ScalarAligner
                            .traceback_batch(&requests[..n], options, &mut scalar)
                            .map_err(|e| e.0)
                    );
                }
            }
        }
        requests[3].word_size = 0;
        let options = AlignOptions {
            max_cells: 1,
            ..Default::default()
        };
        let expected = ScalarAligner
            .traceback_batch(&requests, options, &mut scalar)
            .unwrap_err()
            .0;
        assert!(expected.contains("cell budget"));
        assert_eq!(
            traceback_batch(&requests, options, true, &mut scalar)
                .unwrap_err()
                .0,
            expected
        );
    }

    #[test]
    fn traceback_exhaustive_short_flanks_preserve_ties_and_ambiguity() {
        let words: Vec<Vec<u8>> = (0..=3)
            .flat_map(|len| {
                (0..1usize << len)
                    .map(move |bits| (0..len).map(|i| ((bits >> i) & 1) as u8).collect())
            })
            .collect();
        let mut kernel = Scratch::default();
        let mut scalar = WorkerScratch::default();
        for q in &words {
            for s in &words {
                for reverse in [false, true] {
                    let mut query = q.clone();
                    let mut subject = s.clone();
                    if reverse {
                        query.push(2);
                        subject.push(2);
                    } else {
                        query.insert(0, 2);
                        subject.insert(0, 2);
                    }
                    // One lane has ambiguities; the other three exercise ordinary score ties.
                    let ambiguous: Vec<_> =
                        query.iter().map(|&b| if b == 1 { 4 } else { b }).collect();
                    let requests = std::array::from_fn(|lane| AlignRequest {
                        query: if lane == 3 { &ambiguous } else { &query },
                        subject: &subject,
                        query_anchor: if reverse { q.len() } else { 0 },
                        subject_anchor: if reverse { s.len() } else { 0 },
                        word_size: 1,
                    });
                    for band in [None, Some(0), Some(1)] {
                        for xdrop in [0, 1, 7] {
                            for scoring in [
                                ScoringScheme::default(),
                                ScoringScheme {
                                    reward: 1,
                                    penalty: -1,
                                    gap_open: 0,
                                    gap_extend: 1,
                                },
                            ] {
                                let options = AlignOptions {
                                    band,
                                    xdrop,
                                    scoring,
                                    ..Default::default()
                                };
                                // SAFETY: exact seed base 2, validated options, and aarch64 NEON.
                                let actual = unsafe { trace_four(&requests, options, &mut kernel) }
                                    .unwrap()
                                    .unwrap();
                                assert_eq!(
                                    actual.to_vec(),
                                    ScalarAligner
                                        .traceback_batch(&requests, options, &mut scalar)
                                        .unwrap(),
                                    "q={q:?} s={s:?} reverse={reverse} {options:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn full_i32_scores_and_oversized_window_fallback() {
        let sequence = vec![0; 512];
        let requests = std::array::from_fn(|i| AlignRequest {
            query: &sequence,
            subject: &sequence,
            query_anchor: i * 100,
            subject_anchor: i * 100,
            word_size: 7,
        });
        let options = AlignOptions {
            band: Some(8),
            xdrop: 1000,
            scoring: ScoringScheme {
                reward: 100,
                ..Default::default()
            },
            ..Default::default()
        };
        // SAFETY: valid seeds and options, aarch64 NEON.
        let actual = unsafe { score_four(&requests, options, &mut Scratch::default()) }
            .unwrap()
            .unwrap();
        assert_eq!(
            actual.to_vec(),
            ScalarAligner.score_batch(&requests, options).unwrap()
        );
        assert!(actual.iter().all(|h| h.score == 51_200));
        assert_eq!(
            traceback_batch(&requests, options, true, &mut WorkerScratch::default()).unwrap(),
            ScalarAligner
                .traceback_batch(&requests, options, &mut WorkerScratch::default())
                .unwrap()
        );

        let q = [0; 3];
        let s = vec![0; MAX_COLUMNS + 10];
        let requests = [AlignRequest {
            query: &q,
            subject: &s,
            query_anchor: 0,
            subject_anchor: 0,
            word_size: 1,
        }; 4];
        let options = AlignOptions {
            band: None,
            ..Default::default()
        };
        // SAFETY: valid seeds and options, aarch64 NEON. This must request fallback, not truncate.
        assert!(
            unsafe { score_four(&requests, options, &mut Scratch::default()) }
                .unwrap()
                .is_none()
        );
        assert_eq!(
            score_batch(&requests, options, true).unwrap(),
            ScalarAligner.score_batch(&requests, options).unwrap()
        );
    }
}
