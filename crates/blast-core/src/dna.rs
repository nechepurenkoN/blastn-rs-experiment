use crate::{Error, Result};

/// ACGT occupy two bits; ambiguous symbols and lowercase are preserved separately.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedDna {
    len: usize,
    packed: Vec<u8>,
    ambiguous: Vec<(usize, u8)>,
    lowercase: Vec<bool>,
}

const ENCODE: [u8; 256] = {
    let mut table = [u8::MAX; 256];
    let alphabet = b"ACGTRYSWKMBDHVN";
    let mut i = 0;
    while i < alphabet.len() {
        table[alphabet[i] as usize] = i as u8;
        table[(alphabet[i] + b'a' - b'A') as usize] = i as u8;
        i += 1;
    }
    table
};

pub fn encode(base: u8) -> Option<u8> {
    let code = ENCODE[base as usize];
    (code != u8::MAX).then_some(code)
}
pub fn decode(base: u8) -> u8 {
    b"ACGTRYSWKMBDHVN"[base as usize]
}
pub fn complement(base: u8) -> u8 {
    [3, 2, 1, 0, 5, 4, 6, 7, 9, 8, 13, 12, 11, 10, 14][base as usize]
}
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter().rev().map(|&b| complement(b)).collect()
}

impl PackedDna {
    pub fn from_ascii(text: &[u8]) -> Result<Self> {
        let mut packed = vec![0; text.len().div_ceil(4)];
        let mut ambiguous = Vec::new();
        let mut lowercase = Vec::with_capacity(text.len());
        for (i, &b) in text.iter().enumerate() {
            let code = encode(b).ok_or_else(|| {
                Error(format!("invalid DNA symbol {:?} at offset {i}", b as char))
            })?;
            if code < 4 {
                packed[i / 4] |= code << (2 * (i % 4));
            } else {
                ambiguous.push((i, code));
            }
            lowercase.push(b.is_ascii_lowercase());
        }
        Ok(Self {
            len: text.len(),
            packed,
            ambiguous,
            lowercase,
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn unpack(&self) -> Vec<u8> {
        let mut result: Vec<_> = (0..self.len)
            .map(|i| (self.packed[i / 4] >> (2 * (i % 4))) & 3)
            .collect();
        for &(i, b) in &self.ambiguous {
            result[i] = b;
        }
        result
    }
    pub fn to_ascii(&self) -> Vec<u8> {
        self.unpack()
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                let b = decode(b);
                if self.lowercase[i] {
                    b.to_ascii_lowercase()
                } else {
                    b
                }
            })
            .collect()
    }
    pub fn lowercase_mask(&self) -> &[bool] {
        &self.lowercase
    }
}

#[derive(Clone, Debug)]
pub struct SequenceRecord {
    pub id: String,
    pub description: String,
    pub sequence: PackedDna,
}

/// Search-only subject data: IUPAC codes, without packing or copying FASTA metadata.
pub struct DecodedSubject {
    pub id: String,
    pub sequence: Vec<u8>,
}
#[derive(Clone, Copy)]
pub struct SequenceView<'a> {
    pub id: &'a str,
    pub sequence: &'a [u8],
}
impl DecodedSubject {
    pub fn view(&self) -> SequenceView<'_> {
        SequenceView {
            id: &self.id,
            sequence: &self.sequence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decode_table_matches_original_for_every_byte() {
        for b in 0..=255u8 {
            let expected = b"ACGTRYSWKMBDHVN"
                .iter()
                .position(|&c| c == b.to_ascii_uppercase())
                .map(|i| i as u8);
            assert_eq!(encode(b), expected);
        }
    }
    #[test]
    fn lossless_iupac_roundtrip() {
        let text = b"ACGTRYSWKMBDHVNacgt";
        let seq = PackedDna::from_ascii(text).unwrap();
        assert_eq!(seq.to_ascii(), text);
        assert_eq!(
            reverse_complement(&reverse_complement(&seq.unpack())),
            seq.unpack()
        );
        assert!(PackedDna::from_ascii(b"AC-G").is_err());
    }
}
