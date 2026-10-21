//! Native windowed masking and the symmetric DUST perfect-interval algorithm.

/// Symmetric DUST for unambiguous DNA: window 64, level 20, linker 1.
/// Implements the perfect-interval definition in NCBI's sdust.tex rather than
/// the legacy fixed-window approximation below. NCBI randomizes N during DUST;
/// until that converter is ported, reject ambiguity rather than claim parity.
pub fn symmetric_dust_mask(seq: &[u8]) -> crate::Result<Vec<bool>> {
    if seq.iter().any(|&b| b >= 4) {
        return Err(crate::Error("NCBI DUST compatibility currently requires A/C/G/T queries; use --dust no for ambiguous queries".into()));
    }
    #[derive(Clone, Copy)]
    struct Perfect {
        start: usize,
        end: usize,
        score: usize,
        denominator: usize,
    }
    let mut perfect = Vec::<Perfect>::new(); // descending start, like NCBI's P
    let mut regions = Vec::<(usize, usize)>::new();
    let save = |regions: &mut Vec<(usize, usize)>, p: Perfect| {
        if let Some(last) = regions.last_mut()
            && last.1 + 1 >= p.start
        {
            last.1 = last.1.max(p.end);
        } else {
            regions.push((p.start, p.end));
        }
    };
    for end in 3..seq.len() {
        let start = (end + 1).saturating_sub(64);
        if let Some(&p) = perfect.last()
            && p.start < start
        {
            save(&mut regions, p);
            while perfect.last().is_some_and(|p| p.start < start) {
                perfect.pop();
            }
        }
        let mut counts = [0usize; 64];
        let (mut score, mut index, mut best_score, mut best_len) = (0, 0, 0, 0);
        for pos in (start..=end - 2).rev() {
            let t = ((seq[pos] << 4) | (seq[pos + 1] << 2) | seq[pos + 2]) as usize;
            score += counts[t];
            counts[t] += 1;
            let denominator = end - pos - 2; // number of triplets minus one
            if denominator == 0 || 10 * score <= 20 * denominator {
                continue;
            }
            while index < perfect.len() && perfect[index].start >= pos {
                let p = perfect[index];
                if best_score == 0 || p.score * best_len > best_score * p.denominator {
                    best_score = p.score;
                    best_len = p.denominator;
                }
                index += 1;
            }
            if best_score == 0 || score * best_len >= best_score * denominator {
                best_score = score;
                best_len = denominator;
                perfect.insert(
                    index,
                    Perfect {
                        start: pos,
                        end,
                        score,
                        denominator,
                    },
                );
            }
        }
    }
    while let Some(p) = perfect.pop() {
        save(&mut regions, p);
    }
    let mut mask = vec![false; seq.len()];
    for (start, end) in regions {
        mask[start..=end].fill(true);
    }
    Ok(mask)
}

/// Legacy native mask, retained for backward-compatible native searches.
pub fn dust_mask(seq: &[u8], window: usize, level: f64) -> Vec<bool> {
    let mut mask = vec![false; seq.len()];
    if seq.len() < 3 {
        return mask;
    }
    let width = window.max(3).min(seq.len());
    // Difference array keeps marking many overlapping windows linear in input size.
    let mut delta = vec![0i64; seq.len() + 1];
    let triplet = |i: usize| {
        if seq[i..i + 3].iter().all(|&b| b < 4) {
            Some(((seq[i] << 4) | (seq[i + 1] << 2) | seq[i + 2]) as usize)
        } else {
            None
        }
    };
    let mut counts = [0usize; 64];
    let mut pairs = 0usize;
    for i in 0..width - 2 {
        if let Some(t) = triplet(i) {
            pairs += counts[t];
            counts[t] += 1;
        }
    }
    for start in 0..=seq.len() - width {
        if 10.0 * pairs as f64 / (width - 2) as f64 > level {
            delta[start] += 1;
            delta[start + width] -= 1;
        }
        if start + width < seq.len() {
            if let Some(t) = triplet(start) {
                counts[t] -= 1;
                pairs -= counts[t];
            }
            if let Some(t) = triplet(start + width - 2) {
                pairs += counts[t];
                counts[t] += 1;
            }
        }
    }
    let mut active = 0;
    for (i, m) in mask.iter_mut().enumerate() {
        active += delta[i];
        *m = active > 0;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn symmetric_dust_matches_ncbi_golden_masks() {
        // NCBI BLAST+ 2.17.0 JSON query_masking; inclusive zero-based ranges.
        // Covers short/boundary windows, tandem repeats and reverse complements.
        for line in include_str!("mask/ncbi-2.17.0.tsv").lines() {
            let fields: Vec<_> = line.split('\t').collect();
            let seq = crate::dna::PackedDna::from_ascii(fields[1].as_bytes())
                .unwrap()
                .unpack();
            let mut expected = vec![false; seq.len()];
            if fields[2] != "-" {
                for interval in fields[2].split(',') {
                    let (a, b) = interval.split_once('-').unwrap();
                    expected[a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap()].fill(true);
                }
            }
            assert_eq!(
                symmetric_dust_mask(&seq).unwrap(),
                expected,
                "{}",
                fields[0]
            );
        }
        assert!(symmetric_dust_mask(&[0, 4, 1]).is_err());
        assert_eq!(symmetric_dust_mask(&[]).unwrap(), Vec::<bool>::new());
    }
    #[test]
    fn repetitive_and_tiny() {
        assert!(dust_mask(&[0; 80], 64, 20.0).iter().all(|&x| x));
        assert_eq!(dust_mask(&[0, 1], 64, 20.0), vec![false; 2]);
        assert!(
            !dust_mask(&[0, 1, 2, 3, 0, 2, 1, 3], 64, 20.0)
                .iter()
                .any(|&x| x)
        );
    }
}
