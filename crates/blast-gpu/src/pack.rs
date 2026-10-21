//! Reusable upload storage. A batch is split at a whole request, never at a flank.
use super::gpu_error;
use blast_core::{
    Result,
    align::{AlignOptions, AlignRequest},
};

#[derive(Default)]
pub(crate) struct PackedBatch {
    pub dna: Vec<u8>,
    pub jobs: Vec<u8>,
    pub scratch_bytes: usize,
    pub count: usize,
}

impl PackedBatch {
    pub fn prepare(
        &mut self,
        requests: &[AlignRequest<'_>],
        options: AlignOptions,
        limit: usize,
    ) -> Result<()> {
        self.dna.clear();
        self.jobs.clear();
        self.scratch_bytes = 0;
        self.count = 0;
        for &r in requests {
            r.validate()?;
            let (q, s, k) = (r.query_anchor, r.subject_anchor, r.word_size);
            let left_n = options.band.map_or(s, |b| s.min(q.saturating_add(b)));
            let right_q = r.query.len() - q - k;
            let right_s = r.subject.len() - s - k;
            let right_n = options
                .band
                .map_or(right_s, |b| right_s.min(right_q.saturating_add(b)));
            let dna_bytes = (q + left_n + right_q + right_n).checked_mul(4);
            let scratch_bytes = (left_n + right_n + 2).checked_mul(16);
            let fits = dna_bytes
                .and_then(|n| self.dna.len().checked_add(n))
                .is_some_and(|n| n <= limit)
                && scratch_bytes
                    .and_then(|n| self.scratch_bytes.checked_add(n))
                    .is_some_and(|n| n <= limit)
                && self.jobs.len() + 128 <= limit;
            if !fits {
                if self.count == 0 {
                    return Err(gpu_error(
                        "one alignment exceeds GPU buffer budget; use CPU or a smaller window",
                    ));
                }
                break;
            }
            self.side(&r.query[..q], &r.subject[s - left_n..s], true, options);
            self.side(
                &r.query[q + k..],
                &r.subject[s + k..s + k + right_n],
                false,
                options,
            );
            self.count += 1;
        }
        if self.dna.is_empty() {
            self.dna.extend(0u32.to_le_bytes());
        }
        Ok(())
    }

    fn side(&mut self, query: &[u8], subject: &[u8], reverse: bool, options: AlignOptions) {
        let qoff = self.dna.len() / 4;
        let soff = qoff + query.len();
        for seq in [query, subject] {
            if reverse {
                self.dna
                    .extend(seq.iter().rev().flat_map(|&b| (b as u32).to_le_bytes()));
            } else {
                self.dna
                    .extend(seq.iter().flat_map(|&b| (b as u32).to_le_bytes()));
            }
        }
        let words = [
            qoff as u32,
            query.len() as u32,
            soff as u32,
            subject.len() as u32,
            (self.scratch_bytes / 4) as u32,
            options
                .band
                .unwrap_or(query.len().max(subject.len()))
                .min(u32::MAX as usize) as u32,
            options.xdrop as u32,
            options.max_cells.min(u32::MAX as usize) as u32,
            options.scoring.reward as u32,
            options.scoring.penalty as u32,
            options.scoring.gap_open as u32,
            options.scoring.gap_extend as u32,
            u32::from(options.band.is_some()),
            0,
            0,
            0,
        ];
        self.jobs
            .extend(words.into_iter().flat_map(u32::to_le_bytes));
        self.scratch_bytes += 16 * (subject.len() + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_splits_whole_requests_and_rejects_oversized_single_request() {
        let dna = vec![0; 32];
        let r = AlignRequest {
            query: &dna,
            subject: &dna,
            query_anchor: 8,
            subject_anchor: 8,
            word_size: 4,
        };
        let mut p = PackedBatch::default();
        p.prepare(&[r; 4], AlignOptions::default(), 1000).unwrap();
        assert_eq!(p.count, 2);
        assert_eq!(p.jobs.len(), 256);
        assert_eq!(p.scratch_bytes, 960);
        // Both reversed left flanks precede the two forward right flanks.
        assert_eq!(p.dna.len(), 448);
        assert!(p.prepare(&[r], AlignOptions::default(), 400).is_err());
        p.prepare(&[r], AlignOptions::default(), 1000).unwrap();
        assert_eq!(p.count, 1);
    }
}
