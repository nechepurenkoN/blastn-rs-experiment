//! Parameters transcribed from NCBI blast_stat.c, blastn_values_2_3 / 1_3.
use crate::{Error, Result, align::ScoringScheme};

/// Ungapped nucleotide statistics against the uniform A/C/G/T background used
/// by BLASTN. They must not be replaced by the gapped statistical table.
#[derive(Clone, Copy, Debug)]
pub struct UngappedStats {
    pub lambda: f64,
    pub k: f64,
}
impl UngappedStats {
    pub fn for_scoring(s: ScoringScheme) -> Result<Self> {
        // Keep the supported scoring domain identical to the gapped tables.
        KarlinStats::for_scoring(s)?;
        let (reward, penalty) = (s.reward as f64, s.penalty as f64);
        let (mut lo, mut hi) = (0.0, 2.0);
        for _ in 0..80 {
            let mid = (lo + hi) / 2.0;
            if 0.25 * (mid * reward).exp() + 0.75 * (mid * penalty).exp() > 1.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let lambda = (lo + hi) / 2.0;
        let h_over_lambda =
            0.25 * reward * (lambda * reward).exp() + 0.75 * penalty * (lambda * penalty).exp();
        let k = if s.reward == 1 {
            h_over_lambda * -(-lambda).exp_m1()
        } else {
            // BlastKarlinLHtoK's probability convolution, specialized to the
            // two possible scores (2/-3, gcd=1). Same 100/0.0001 stopping rule.
            let mut probabilities = vec![1.0];
            let mut outer = 0.0;
            for n in 1..=100 {
                let mut next = vec![0.0; probabilities.len() + 1];
                for (matches, &p) in probabilities.iter().enumerate() {
                    next[matches] += 0.75 * p;
                    next[matches + 1] += 0.25 * p;
                }
                let inner: f64 = next
                    .iter()
                    .enumerate()
                    .map(|(matches, &p)| {
                        let score = matches as f64 * reward + (n - matches) as f64 * penalty;
                        p * if score < 0.0 {
                            (lambda * score).exp()
                        } else {
                            1.0
                        }
                    })
                    .sum();
                let term = inner / n as f64;
                outer += term;
                probabilities = next;
                if term <= 0.0001 {
                    break;
                }
            }
            (-2.0 * outer).exp() / (h_over_lambda * -(-lambda).exp_m1())
        };
        Ok(Self { lambda, k })
    }
    pub fn raw_drop(self, bits: f64) -> i32 {
        (bits * std::f64::consts::LN_2 / self.lambda).ceil() as i32
    }
    pub fn gap_trigger(self, bits: f64) -> i32 {
        ((bits * std::f64::consts::LN_2 + self.k.ln()) / self.lambda) as i32
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KarlinStats {
    pub lambda: f64,
    pub k: f64,
    pub alpha: f64,
    pub beta: f64,
    pub round_even: bool,
}
impl KarlinStats {
    pub fn for_scoring(s: ScoringScheme) -> Result<Self> {
        let (lambda,k,alpha,beta,round_even) = match (s.reward,s.penalty,s.gap_open,s.gap_extend) {
            (2,-3,5,2) => (0.625,0.41,0.8,-2.0,true),
            (2,-3,4,2) => (0.61,0.35,0.9,-3.0,true),
            (2,-3,2,2) => (0.515,0.14,1.55,-9.0,true),
            (2,-3,6,2) | (2,-3,4,4) => (0.63,0.42,0.75,-2.0,true),
            (2,-3,2,4) => (0.615,0.37,0.85,-3.0,true),
            (2,-3,3,3) => (0.615,0.37,0.9,-3.0,true),
            (2,-3,0,4) => (0.55,0.21,1.2,-5.0,true),
            (1,-3,2,2) => (1.37,0.70,1.1,0.0,false),
            (1,-3,1,2) => (1.35,0.64,1.2,-1.0,false),
            (1,-3,0,2) => (1.25,0.42,1.5,-2.0,false),
            (1,-3,2,1) => (1.34,0.60,1.2,-1.0,false),
            (1,-3,1,1) => (1.21,0.34,1.7,-2.0,false),
            _ => return Err(Error("no validated statistical table for this scoring scheme; supported reward/penalty: 2/-3 or 1/-3 with documented gap costs".into())),
        };
        Ok(Self {
            lambda,
            k,
            alpha,
            beta,
            round_even,
        })
    }
    pub fn bits(self, score: i32) -> f64 {
        (self.lambda * score as f64 - self.k.ln()) / std::f64::consts::LN_2
    }
    pub fn raw_drop(self, bits: f64) -> i32 {
        (bits * std::f64::consts::LN_2 / self.lambda).ceil() as i32
    }
    /// BLAST_Cutoffs for a single HSP, before even-score E-value rounding.
    pub fn score_cutoff(self, evalue: f64, search_space: f64) -> i32 {
        if evalue == 0.0 {
            return i32::MAX;
        }
        ((self.k * search_space / evalue).ln() / self.lambda)
            .ceil()
            .max(1.0) as i32
    }
    pub fn evalue(self, score: i32, search_space: f64) -> f64 {
        let s = if self.round_even {
            score - score.rem_euclid(2)
        } else {
            score
        };
        self.k * search_space * (-self.lambda * s as f64).exp()
    }
    pub fn effective_space(self, query_len: usize, db_len: u64, db_count: usize) -> f64 {
        let (m, n, count) = (query_len as f64, db_len as f64, db_count as f64);
        if m == 0.0 || n == 0.0 || count == 0.0 {
            return 0.0;
        }
        // Same upper bound and fixed-point inequality as BLAST_ComputeLengthAdjustment.
        let c = m * n - m.max(n) / self.k;
        if c <= 0.0 {
            return m * n;
        }
        let mb = m * count + n;
        let upper = 2.0 * c / (mb + (mb * mb - 4.0 * count * c).max(0.0).sqrt());
        let mut lo = 0u64;
        let mut hi = upper.floor().max(0.0) as u64;
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            let ell = mid as f64;
            let f = self.alpha / self.lambda * (self.k.ln() + ((m - ell) * (n - count * ell)).ln())
                + self.beta;
            if f >= ell {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let ell = lo as f64;
        (m - ell).max(1.0) * (n - count * ell).max(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ncbi_tables_for_every_supported_gap_scheme() {
        // Independent rows from NCBI blast_stat.c, blastn_values_2_3 / 1_3.
        // Columns: gap open, gap extend, lambda, K, H, alpha, beta, theta.
        // In particular, H=1.1 is NOT alpha=1.2 for 1/-3 with gaps 2/1.
        let tables: &[(i32, i32, bool, &[[f64; 8]])] = &[
            (
                2,
                -3,
                true,
                &[
                    [4., 4., 0.63, 0.42, 0.84, 0.75, -2., 99.],
                    [2., 4., 0.615, 0.37, 0.72, 0.85, -3., 97.],
                    [0., 4., 0.55, 0.21, 0.46, 1.2, -5., 87.],
                    [3., 3., 0.615, 0.37, 0.68, 0.9, -3., 97.],
                    [6., 2., 0.63, 0.42, 0.84, 0.75, -2., 99.],
                    [5., 2., 0.625, 0.41, 0.78, 0.8, -2., 99.],
                    [4., 2., 0.61, 0.35, 0.68, 0.9, -3., 96.],
                    [2., 2., 0.515, 0.14, 0.33, 1.55, -9., 81.],
                ],
            ),
            (
                1,
                -3,
                false,
                &[
                    [2., 2., 1.37, 0.70, 1.2, 1.1, 0., 99.],
                    [1., 2., 1.35, 0.64, 1.1, 1.2, -1., 98.],
                    [0., 2., 1.25, 0.42, 0.83, 1.5, -2., 91.],
                    [2., 1., 1.34, 0.60, 1.1, 1.2, -1., 97.],
                    [1., 1., 1.21, 0.34, 0.71, 1.7, -2., 88.],
                ],
            ),
        ];
        for &(reward, penalty, round_even, rows) in tables {
            for row in rows {
                let stats = KarlinStats::for_scoring(ScoringScheme {
                    reward,
                    penalty,
                    gap_open: row[0] as i32,
                    gap_extend: row[1] as i32,
                })
                .unwrap();
                assert_eq!(
                    [stats.lambda, stats.k, stats.alpha, stats.beta],
                    [row[2], row[3], row[5], row[6]]
                );
                assert_eq!(stats.round_even, round_even);
            }
        }
    }

    #[test]
    fn gap_2_1_effective_space_keeps_hit_below_evalue_cutoff() {
        let stats = KarlinStats::for_scoring(ScoringScheme {
            reward: 1,
            penalty: -3,
            gap_open: 2,
            gap_extend: 1,
        })
        .unwrap();
        // NCBI length adjustment is 14: (50-14) * (1_000_000-100*14).
        let space = stats.effective_space(50, 1_000_000, 100);
        assert_eq!(space, 35_949_600.0);
        assert!((stats.evalue(20, space) - 4.9516871490926656e-5).abs() < 1e-16);
        assert!(stats.evalue(20, space) < 5e-5);
    }

    #[test]
    fn ungapped_statistics_drive_ncbi_trigger_and_xdrop() {
        let u = UngappedStats::for_scoring(ScoringScheme::default()).unwrap();
        assert!((u.lambda - 0.6337314431).abs() < 1e-9);
        assert!((u.k - 0.408).abs() < 0.001);
        assert_eq!(u.raw_drop(20.0), 22);
        assert_eq!(u.gap_trigger(27.0), 28);
        let g = KarlinStats::for_scoring(ScoringScheme::default()).unwrap();
        assert_eq!(g.raw_drop(20.0), 23); // original compatibility bug
        assert_eq!(g.score_cutoff(10.0, 6935936930.0), 32);
        assert_eq!(g.score_cutoff(10.0, 100.0), 3);
        assert_eq!(g.score_cutoff(0.0, 100.0), i32::MAX);
        let u = UngappedStats::for_scoring(ScoringScheme {
            reward: 1,
            gap_open: 2,
            ..Default::default()
        })
        .unwrap();
        assert!((u.lambda - 1.374063122).abs() < 1e-8);
        assert!((u.k - 0.711).abs() < 0.001);
        assert_eq!(u.raw_drop(20.0), 11);
        assert_eq!(u.gap_trigger(27.0), 13);
    }
    #[test]
    fn known_table_and_rounding() {
        let stats = KarlinStats::for_scoring(ScoringScheme::default()).unwrap();
        assert_eq!(stats.lambda, 0.625);
        assert_eq!(stats.evalue(31, 1000.0), stats.evalue(30, 1000.0));
        assert!(stats.bits(31) > stats.bits(30));
        let space = stats.effective_space(100, 1_000_000, 100);
        assert!(space > 0.0 && space < 100_000_000.0);
        assert_eq!(stats.effective_space(0, 100, 1), 0.0);
    }
}
