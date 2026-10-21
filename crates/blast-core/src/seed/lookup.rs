//! NCBI-style backbone/presence-vector lookup, with exact full-word verification.
//! Unlike NCBI's short-word scanners, this keeps the original stride-one seed stream.
use std::ops::Range;

const MAX_BUCKETS: usize = 1 << 16;

pub(crate) struct WordRange {
    pub word: u64,
    pub positions: Range<usize>,
}

pub(crate) enum WordLookup {
    Empty,
    Backbone {
        mask: u64,
        present: Vec<u64>,
        offsets: Vec<u32>,
        words: Vec<WordRange>,
    },
    // Avoid narrowing overflow for exceptionally large library-side query batches.
    Sorted(Vec<WordRange>),
}

impl WordLookup {
    /// `words` must contain unique full words and ranges into an unchanged postings array.
    pub fn new(word_size: usize, words: Vec<WordRange>) -> Self {
        Self::with_index_limit(word_size, words, u32::MAX as usize)
    }

    fn with_index_limit(word_size: usize, mut words: Vec<WordRange>, index_limit: usize) -> Self {
        if words.is_empty() {
            return Self::Empty;
        }
        if words.len() > index_limit {
            words.sort_unstable_by_key(|entry| entry.word);
            return Self::Sorted(words);
        }
        let buckets = words
            .len()
            .saturating_mul(8)
            .clamp(16, MAX_BUCKETS)
            .next_power_of_two()
            .min(1 << word_size.saturating_mul(2).min(16));
        let mask = (buckets - 1) as u64;
        words.sort_unstable_by_key(|entry| (entry.word & mask, entry.word));
        let mut offsets = vec![0u32; buckets + 1];
        let mut present = vec![0u64; buckets.div_ceil(64)];
        for entry in &words {
            let bucket = (entry.word & mask) as usize;
            offsets[bucket + 1] += 1;
            present[bucket / 64] |= 1 << (bucket % 64);
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        Self::Backbone {
            mask,
            present,
            offsets,
            words,
        }
    }

    #[inline]
    pub fn get(&self, word: u64) -> Option<Range<usize>> {
        let entries = match self {
            Self::Empty => return None,
            Self::Sorted(entries) => entries.as_slice(),
            Self::Backbone {
                mask,
                present,
                offsets,
                words,
            } => {
                let bucket = (word & mask) as usize;
                if present[bucket / 64] & (1 << (bucket % 64)) == 0 {
                    return None;
                }
                &words[offsets[bucket] as usize..offsets[bucket + 1] as usize]
            }
        };
        // A prefix collision must never become an additional seed. Long buckets use
        // binary search so repetitive/adversarial words cannot cause linear scans.
        let entry = if entries.len() <= 4 {
            entries.iter().find(|entry| entry.word == word)?
        } else {
            &entries[entries
                .binary_search_by_key(&word, |entry| entry.word)
                .ok()?]
        };
        Some(entry.positions.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn exhaustive_small_words_match_independent_hash_lookup() {
        for k in 1..=6 {
            let expected: HashMap<_, _> = (0..1u64 << (2 * k))
                .filter(|w| w % 3 == 0)
                .map(|word| (word, word as usize * 2..word as usize * 2 + 2))
                .collect();
            let lookup = WordLookup::new(
                k,
                expected
                    .iter()
                    .map(|(&word, positions)| WordRange {
                        word,
                        positions: positions.clone(),
                    })
                    .collect(),
            );
            for word in 0..1u64 << (2 * k + 1) {
                assert_eq!(
                    lookup.get(word),
                    expected.get(&word).cloned(),
                    "k={k} word={word}"
                );
            }
            assert_eq!(lookup.get(u64::MAX), None);
        }
    }

    #[test]
    fn full_width_collisions_empty_and_narrowing_fallback() {
        let mut keys: Vec<_> = (0..257u64).map(|i| i << 32 | 0xabcd).collect();
        keys.extend([0, u64::MAX, 1 << 63]);
        let expected: HashMap<_, _> = keys
            .iter()
            .enumerate()
            .map(|(i, &word)| (word, i * 3..i * 3 + 3))
            .collect();
        for limit in [16, u32::MAX as usize] {
            let lookup = WordLookup::with_index_limit(
                32,
                expected
                    .iter()
                    .map(|(&word, positions)| WordRange {
                        word,
                        positions: positions.clone(),
                    })
                    .collect(),
                limit,
            );
            for (&word, positions) in &expected {
                assert_eq!(lookup.get(word), Some(positions.clone()));
            }
            for word in [1, 258 << 32 | 0xabcd, u64::MAX - 1] {
                assert_eq!(lookup.get(word), None);
            }
        }
        assert!(matches!(WordLookup::new(32, vec![]), WordLookup::Empty));
        assert_eq!(WordLookup::new(32, vec![]).get(0), None);
    }
}
