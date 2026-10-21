use crate::{Error, Result};
use std::collections::HashMap;

pub(crate) mod lookup;

pub struct Kmers<'a> {
    seq: &'a [u8],
    mask: Option<&'a [bool]>,
    k: usize,
    position: usize,
    valid: usize,
    key: u64,
    bits: u64,
}
impl<'a> Kmers<'a> {
    pub fn new(seq: &'a [u8], mask: Option<&'a [bool]>, k: usize) -> Result<Self> {
        if !(1..=32).contains(&k) {
            return Err(Error("word size must be 1..=32".into()));
        }
        if mask.is_some_and(|m| m.len() != seq.len()) {
            return Err(Error("mask length differs from sequence".into()));
        }
        Ok(Self {
            seq,
            mask,
            k,
            position: 0,
            valid: 0,
            key: 0,
            bits: if k == 32 {
                u64::MAX
            } else {
                (1 << (2 * k)) - 1
            },
        })
    }
}
impl Iterator for Kmers<'_> {
    type Item = (usize, u64);
    fn next(&mut self) -> Option<Self::Item> {
        while self.position < self.seq.len() {
            let i = self.position;
            self.position += 1;
            let b = self.seq[i];
            if b > 3 || self.mask.is_some_and(|m| m[i]) {
                self.valid = 0;
                self.key = 0;
                continue;
            }
            self.key = ((self.key << 2) | b as u64) & self.bits;
            self.valid += 1;
            if self.valid >= self.k {
                return Some((i + 1 - self.k, self.key));
            }
        }
        None
    }
}

pub trait SeedIndex: Sync {
    fn lookup(&self, key: u64) -> &[usize];
}

/// Query-side index: compact contiguous postings, with O(1) key lookup.
pub struct QueryLookup {
    offsets: HashMap<u64, (usize, usize)>,
    positions: Vec<usize>,
    pub suppressed_words: usize,
}
impl QueryLookup {
    /// Already-masked, frequency-filtered postings. Positions retain ascending query order.
    pub fn entries(&self) -> impl Iterator<Item = (u64, &[usize])> {
        self.offsets
            .iter()
            .map(|(&key, &(a, b))| (key, &self.positions[a..b]))
    }
    pub fn build(seq: &[u8], mask: &[bool], k: usize, max_occurrences: usize) -> Result<Self> {
        let mut words: Vec<_> = Kmers::new(seq, Some(mask), k)?
            .map(|(p, key)| (key, p))
            .collect();
        words.sort_unstable();
        let mut result = Self {
            offsets: HashMap::new(),
            positions: Vec::with_capacity(words.len()),
            suppressed_words: 0,
        };
        let mut i = 0;
        while i < words.len() {
            let mut end = i + 1;
            while end < words.len() && words[end].0 == words[i].0 {
                end += 1;
            }
            if max_occurrences == 0 || end - i <= max_occurrences {
                let start = result.positions.len();
                result.positions.extend(words[i..end].iter().map(|x| x.1));
                result
                    .offsets
                    .insert(words[i].0, (start, result.positions.len()));
            } else {
                result.suppressed_words += 1;
            }
            i = end;
        }
        Ok(result)
    }
}
impl SeedIndex for QueryLookup {
    fn lookup(&self, key: u64) -> &[usize] {
        self.offsets
            .get(&key)
            .map_or(&[], |&(a, b)| &self.positions[a..b])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ambiguity_mask_and_32mer() {
        assert_eq!(
            Kmers::new(&[0, 1, 14, 2, 3], None, 2)
                .unwrap()
                .collect::<Vec<_>>(),
            vec![(0, 1), (3, 11)]
        );
        assert_eq!(
            Kmers::new(&[3; 32], None, 32).unwrap().next(),
            Some((0, u64::MAX))
        );
        assert_eq!(Kmers::new(&[0; 2], None, 3).unwrap().count(), 0);
        assert_eq!(
            Kmers::new(&[0; 3], Some(&[false, true, false]), 2)
                .unwrap()
                .count(),
            0
        );
    }
}
