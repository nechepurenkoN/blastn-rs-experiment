//! FASTA streaming and a portable, versioned nucleotide store (.rdb).
use blast_core::{
    Error, Result,
    dna::{DecodedSubject, PackedDna, SequenceRecord, encode},
};
use memmap2::Mmap;
use std::{
    collections::HashSet,
    fs::{File, OpenOptions},
    io::{BufRead, BufWriter, Write},
    path::Path,
};

pub struct FastaReader<R: BufRead> {
    reader: R,
    pending: Option<String>,
    line: usize,
    seen: HashSet<String>,
    finished: bool,
}
impl<R: BufRead> FastaReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            pending: None,
            line: 0,
            seen: HashSet::new(),
            finished: false,
        }
    }
    fn record(&mut self) -> Result<Option<SequenceRecord>> {
        if self.finished {
            return Ok(None);
        }
        let mut header = self.pending.take();
        let mut data = Vec::new();
        let mut line = String::new();
        loop {
            line.clear();
            if self.reader.read_line(&mut line)? == 0 {
                self.finished = true;
                break;
            }
            self.line += 1;
            let text = line.trim();
            if text.is_empty() {
                continue;
            }
            if let Some(h) = text.strip_prefix('>') {
                if header.is_some() {
                    self.pending = Some(h.to_string());
                    break;
                }
                header = Some(h.to_string());
            } else {
                if header.is_none() {
                    return Err(Error(format!(
                        "FASTA line {}: sequence before header",
                        self.line
                    )));
                }
                data.extend(text.bytes().filter(|b| !b.is_ascii_whitespace()));
                if data.len() > 10_000_000 {
                    return Err(Error("FASTA record exceeds current 10 Mb limit; split long references into contigs".into()));
                }
            }
        }
        let Some(description) = header else {
            return Ok(None);
        };
        let id = description
            .split_whitespace()
            .next()
            .ok_or_else(|| Error("empty FASTA identifier".into()))?
            .to_string();
        if !self.seen.insert(id.clone()) {
            return Err(Error(format!("duplicate FASTA identifier: {id}")));
        }
        if data.is_empty() {
            return Err(Error(format!("empty FASTA sequence: {id}")));
        }
        let sequence =
            PackedDna::from_ascii(&data).map_err(|e| Error(format!("FASTA {id}: {e}")))?;
        Ok(Some(SequenceRecord {
            id,
            description,
            sequence,
        }))
    }
}
impl<R: BufRead> Iterator for FastaReader<R> {
    type Item = Result<SequenceRecord>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.record() {
            Ok(Some(r)) => Some(Ok(r)),
            Ok(None) => None,
            Err(e) => {
                self.finished = true;
                Some(Err(e))
            }
        }
    }
}

pub trait SequenceStore: Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn total_bases(&self) -> u64;
    fn record(&self, index: usize) -> Result<SequenceRecord>;
    fn decoded_record(&self, index: usize) -> Result<DecodedSubject> {
        let record = self.record(index)?;
        Ok(DecodedSubject {
            id: record.id,
            sequence: record.sequence.unpack(),
        })
    }
}
pub struct MemoryStore {
    records: Vec<SequenceRecord>,
    bases: u64,
}
impl MemoryStore {
    pub fn new(records: Vec<SequenceRecord>) -> Self {
        let bases = records.iter().map(|r| r.sequence.len() as u64).sum();
        Self { records, bases }
    }
}
impl SequenceStore for MemoryStore {
    fn len(&self) -> usize {
        self.records.len()
    }
    fn total_bases(&self) -> u64 {
        self.bases
    }
    fn record(&self, index: usize) -> Result<SequenceRecord> {
        self.records
            .get(index)
            .cloned()
            .ok_or_else(|| Error("database record out of bounds".into()))
    }
    fn decoded_record(&self, index: usize) -> Result<DecodedSubject> {
        let record = self
            .records
            .get(index)
            .ok_or_else(|| Error("database record out of bounds".into()))?;
        Ok(DecodedSubject {
            id: record.id.clone(),
            sequence: record.sequence.unpack(),
        })
    }
}

const MAGIC: &[u8; 8] = b"RBLAST01";
fn invalid() -> Error {
    Error("invalid or truncated RBLAST01 database".into())
}
fn u64_at(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    let end = pos.checked_add(8).ok_or_else(invalid)?;
    let data = bytes.get(*pos..end).ok_or_else(invalid)?;
    *pos = end;
    Ok(u64::from_le_bytes(data.try_into().unwrap()))
}
fn usize_at(bytes: &[u8], pos: &mut usize) -> Result<usize> {
    usize::try_from(u64_at(bytes, pos)?).map_err(|_| invalid())
}

struct RecordOffset {
    start: usize,
    header_len: usize,
    len: usize,
}
pub struct BlastDatabase {
    map: Mmap,
    offsets: Vec<RecordOffset>,
    bases: u64,
}
impl BlastDatabase {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: read-only mapping, validated before indexing. The database must not
        // be externally modified/truncated while open (same contract as other mmap readers).
        let map = unsafe { Mmap::map(&file)? };
        if map.get(..8) != Some(MAGIC) {
            return Err(invalid());
        }
        let mut cursor = 8;
        let count = usize_at(&map, &mut cursor)?;
        let bases = u64_at(&map, &mut cursor)?;
        if count > map.len() / 16 {
            return Err(invalid());
        }
        let mut offsets = Vec::with_capacity(count);
        let mut total = 0u64;
        let mut ids = HashSet::new();
        for _ in 0..count {
            let header_len = usize_at(&map, &mut cursor)?;
            let len = usize_at(&map, &mut cursor)?;
            if len == 0 || len > 10_000_000 || header_len == 0 {
                return Err(invalid());
            }
            let end = cursor
                .checked_add(header_len)
                .and_then(|x| x.checked_add(len))
                .ok_or_else(invalid)?;
            if end > map.len() {
                return Err(invalid());
            }
            let description =
                std::str::from_utf8(&map[cursor..cursor + header_len]).map_err(|_| invalid())?;
            let id = description.split_whitespace().next().ok_or_else(invalid)?;
            if !ids.insert(id.to_string()) {
                return Err(Error(format!("duplicate database identifier: {id}")));
            }
            if map[cursor + header_len..end]
                .iter()
                .any(|&b| blast_core::dna::encode(b).is_none())
            {
                return Err(invalid());
            }
            offsets.push(RecordOffset {
                start: cursor,
                header_len,
                len,
            });
            total = total.checked_add(len as u64).ok_or_else(invalid)?;
            cursor = end;
        }
        if cursor != map.len() || total != bases || count == 0 {
            return Err(invalid());
        }
        Ok(Self {
            map,
            offsets,
            bases,
        })
    }
}
impl SequenceStore for BlastDatabase {
    fn len(&self) -> usize {
        self.offsets.len()
    }
    fn total_bases(&self) -> u64 {
        self.bases
    }
    fn decoded_record(&self, index: usize) -> Result<DecodedSubject> {
        let r = self
            .offsets
            .get(index)
            .ok_or_else(|| Error("database record out of bounds".into()))?;
        let end = r.start + r.header_len;
        let description = std::str::from_utf8(&self.map[r.start..end]).map_err(|_| invalid())?;
        let sequence = self.map[end..end + r.len]
            .iter()
            .map(|&b| encode(b).ok_or_else(invalid))
            .collect::<Result<Vec<_>>>()?;
        Ok(DecodedSubject {
            id: description
                .split_whitespace()
                .next()
                .ok_or_else(invalid)?
                .to_owned(),
            sequence,
        })
    }
    fn record(&self, index: usize) -> Result<SequenceRecord> {
        let r = self
            .offsets
            .get(index)
            .ok_or_else(|| Error("database record out of bounds".into()))?;
        let end = r.start + r.header_len;
        let description = std::str::from_utf8(&self.map[r.start..end])
            .map_err(|_| invalid())?
            .to_string();
        Ok(SequenceRecord {
            id: description.split_whitespace().next().unwrap().to_string(),
            description,
            sequence: PackedDna::from_ascii(&self.map[end..end + r.len])?,
        })
    }
}

pub struct DatabaseBuilder;
impl DatabaseBuilder {
    /// Refuses to overwrite. Stage into a sibling temporary file and publish atomically.
    pub fn build(
        records: impl Iterator<Item = Result<SequenceRecord>>,
        path: impl AsRef<Path>,
    ) -> Result<(usize, u64)> {
        let path = path.as_ref();
        if path.exists() {
            return Err(Error(format!("output already exists: {}", path.display())));
        }
        let temporary = path.with_extension(format!("rdb.tmp-{}", std::process::id()));
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&temporary)?;
        let operation = (|| -> Result<(usize, u64)> {
            use std::io::{Seek, SeekFrom};
            let mut writer = BufWriter::new(file);
            writer.write_all(MAGIC)?;
            writer.write_all(&[0; 16])?;
            let (mut count, mut bases) = (0usize, 0u64);
            let mut ids = HashSet::new();
            for record in records {
                let r = record?;
                if !ids.insert(r.id.clone()) {
                    return Err(Error(format!("duplicate identifier: {}", r.id)));
                }
                if r.description.split_whitespace().next() != Some(r.id.as_str())
                    || r.sequence.is_empty()
                    || r.sequence.len() > 10_000_000
                {
                    return Err(Error("invalid database record".into()));
                }
                writer.write_all(&(r.description.len() as u64).to_le_bytes())?;
                writer.write_all(&(r.sequence.len() as u64).to_le_bytes())?;
                writer.write_all(r.description.as_bytes())?;
                writer.write_all(&r.sequence.to_ascii())?;
                count += 1;
                bases += r.sequence.len() as u64;
            }
            if count == 0 {
                return Err(Error("cannot create an empty database".into()));
            }
            writer.seek(SeekFrom::Start(8))?;
            writer.write_all(&(count as u64).to_le_bytes())?;
            writer.write_all(&bases.to_le_bytes())?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
            // hard_link is an atomic no-clobber publication on the same filesystem.
            std::fs::hard_link(&temporary, path)?;
            Ok((count, bases))
        })();
        let _ = std::fs::remove_file(&temporary);
        operation
    }
}

pub fn write_fasta(record: &SequenceRecord, writer: &mut impl Write) -> Result<()> {
    writeln!(writer, ">{}", record.description)?;
    for line in record.sequence.to_ascii().chunks(80) {
        writer.write_all(line)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Read};
    #[test]
    fn fasta_validation() {
        let records = FastaReader::new(Cursor::new(b"\r\n>a description\r\nacgN\r\n\n>b\nTT\n"))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "a");
        assert_eq!(records[0].sequence.to_ascii(), b"acgN");
        for text in [">a\nA\n>a\nT", ">\nAC", ">a\n", "ACGT", ">a\nAC-"] {
            assert!(
                FastaReader::new(Cursor::new(text))
                    .collect::<Result<Vec<_>>>()
                    .is_err(),
                "{text}"
            );
        }
    }
    #[test]
    fn database_roundtrip_and_corruption() {
        let path =
            std::env::temp_dir().join(format!("blast-rs-db-test-{}.rdb", std::process::id()));
        let data = b">q description\nacgtNRYS\n>s\nTTAA\n";
        DatabaseBuilder::build(FastaReader::new(Cursor::new(data)), &path).unwrap();
        let db = BlastDatabase::open(&path).unwrap();
        assert_eq!(db.len(), 2);
        assert_eq!(db.total_bases(), 12);
        assert_eq!(db.record(0).unwrap().sequence.to_ascii(), b"acgtNRYS");
        let memory = MemoryStore::new(
            FastaReader::new(Cursor::new(data))
                .collect::<Result<Vec<_>>>()
                .unwrap(),
        );
        for i in 0..db.len() {
            let expected = db.record(i).unwrap();
            for decoded in [
                db.decoded_record(i).unwrap(),
                memory.decoded_record(i).unwrap(),
            ] {
                assert_eq!(decoded.id, expected.id);
                assert_eq!(decoded.sequence, expected.sequence.unpack());
            }
        }
        assert!(db.decoded_record(db.len()).is_err());
        assert!(memory.decoded_record(memory.len()).is_err());
        assert!(DatabaseBuilder::build(FastaReader::new(Cursor::new(data)), &path).is_err());
        drop(db);
        let mut bytes = Vec::new();
        File::open(&path).unwrap().read_to_end(&mut bytes).unwrap();
        bytes.pop();
        std::fs::write(&path, bytes).unwrap();
        assert!(BlastDatabase::open(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
