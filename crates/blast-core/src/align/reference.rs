//! Frozen pre-optimization implementation, used only as a differential test oracle.
use super::{AlignOptions, AlignRequest, Alignment, AlignmentScore};
use crate::{Error, Result};

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
    cells: Vec<u8>,
}

/// Seed-anchored affine DP, with no local restart. Cost of length L gap = open + L*extend.
/// Rolling score rows and one byte of traceback state per visited cell.
fn extend_side(
    query: &[u8],
    subject: &[u8],
    options: AlignOptions,
    traceback: bool,
) -> Result<Side> {
    let score = options.scoring;
    let band = options.band.unwrap_or(query.len().max(subject.len()));
    let n = subject.len().min(query.len().saturating_add(band));
    let subject = &subject[..n];
    let open = score.gap_open + score.gap_extend;
    let mut prev = vec![NEG; n + 1];
    let mut curr = vec![NEG; n + 1];
    let mut prev_gap = vec![NEG; n + 1];
    let mut curr_gap = vec![NEG; n + 1];
    prev[0] = 0;
    let mut prev_start = 0;
    let mut prev_end = n.min(band);
    for (j, h) in prev.iter_mut().enumerate().take(prev_end + 1).skip(1) {
        let value = -score.gap_open - score.gap_extend * j as i32;
        if value >= -options.xdrop {
            *h = value;
        }
    }
    let mut trace = Vec::<TraceRow>::new();
    if traceback {
        trace.push(TraceRow {
            start: 0,
            cells: vec![3 | 8; prev_end + 1],
        });
    }
    let mut best = Side::default();
    let mut visited = 0usize;
    for i in 1..=query.len() {
        let start = i.saturating_sub(band);
        let end = n.min(i.saturating_add(band));
        if start > end {
            break;
        }
        visited += end - start + 1;
        if visited > options.max_cells {
            return Err(Error(
                "alignment cell budget exceeded; increase --max-cells or reduce --band".into(),
            ));
        }
        let mut trace_cells = if traceback {
            vec![0u8; end - start + 1]
        } else {
            Vec::new()
        };
        let mut horizontal = NEG;
        let mut left_h = NEG;
        let mut alive = false;
        for j in start..=end {
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
                h = diag + score.substitution(query[i - 1], subject[j - 1]);
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
                alive = true;
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
                trace_cells[j - start] = code;
            }
        }
        if traceback {
            trace.push(TraceRow {
                start,
                cells: trace_cells,
            });
        }
        std::mem::swap(&mut prev, &mut curr);
        std::mem::swap(&mut prev_gap, &mut curr_gap);
        prev_start = start;
        prev_end = end;
        if !alive {
            break;
        }
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
            let code = row.cells[j - row.start];
            if state == 0 {
                state = code & 3;
            }
            match state {
                1 => {
                    best.ops
                        .push(if query[i - 1] < 4 && query[i - 1] == subject[j - 1] {
                            b'='
                        } else {
                            b'X'
                        });
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
    request.validate()?;
    options.validate()?;
    let q = request.query_anchor;
    let s = request.subject_anchor;
    let k = request.word_size;
    let qleft: Vec<_> = request.query[..q].iter().copied().rev().collect();
    let sleft_len = options.band.map_or(s, |b| s.min(q.saturating_add(b)));
    let sleft: Vec<_> = request.subject[s - sleft_len..s]
        .iter()
        .copied()
        .rev()
        .collect();
    let left = extend_side(&qleft, &sleft, options, traceback)?;
    let right = extend_side(
        &request.query[q + k..],
        &request.subject[s + k..],
        options,
        traceback,
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
