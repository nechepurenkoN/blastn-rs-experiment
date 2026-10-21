use blast_core::{
    Error, Result,
    search::{Hsp, Strand},
};
use std::io::Write;

#[derive(Clone, Copy)]
pub enum OutputFormat {
    Tabular,
    Commented,
    Json,
}
pub trait ResultFormatter {
    fn write_query(&mut self, query_id: &str, hits: &[Hsp]) -> Result<()>;
    fn finish(&mut self) -> Result<()>;
}
pub struct Formatter<W: Write> {
    writer: W,
    format: OutputFormat,
    first: bool,
    ncbi_numbers: bool,
}
impl<W: Write> Formatter<W> {
    pub fn new(mut writer: W, format: OutputFormat) -> Result<Self> {
        if matches!(format, OutputFormat::Json) {
            writer.write_all(b"[\n")?;
        }
        Ok(Self {
            writer,
            format,
            first: true,
            ncbi_numbers: false,
        })
    }
    /// BLAST+ tabular numeric formatting; JSON retains full-precision numbers.
    pub fn with_ncbi_numbers(mut self, enabled: bool) -> Self {
        self.ncbi_numbers = enabled;
        self
    }
}

fn scientific(value: f64, precision: usize) -> String {
    let text = format!("{value:.precision$e}");
    let (mantissa, exponent) = text.split_once('e').expect("finite scientific number");
    format!("{mantissa}e{:+03}", exponent.parse::<i32>().unwrap())
}
fn ncbi_evalue(value: f64) -> String {
    if value < 1e-180 {
        "0.0".into()
    } else if value < 0.0009 {
        scientific(value, 2)
    } else if value < 0.1 {
        format!("{value:.3}")
    } else if value < 1.0 {
        format!("{value:.2}")
    } else if value < 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.0}")
    }
}
fn ncbi_bits(value: f64) -> String {
    if value > 99999.0 {
        scientific(value, 3)
    } else if value > 99.9 {
        format!("{}", value as u64)
    } else {
        format!("{value:.1}")
    }
}

struct Evalue(f64, bool);
impl std::fmt::Display for Evalue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.1 {
            f.write_str(&ncbi_evalue(self.0))
        } else {
            write!(f, "{:.3e}", self.0)
        }
    }
}
struct BitScore(f64, bool);
impl std::fmt::Display for BitScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.1 {
            f.write_str(&ncbi_bits(self.0))
        } else {
            write!(f, "{:.1}", self.0)
        }
    }
}

impl<W: Write> ResultFormatter for Formatter<W> {
    fn write_query(&mut self, query_id: &str, hits: &[Hsp]) -> Result<()> {
        if matches!(self.format, OutputFormat::Commented) {
            writeln!(
                self.writer,
                "# blastn-rs 0.1.0\n# Query: {query_id}\n# Fields: query id, subject id, % identity, alignment length, mismatches, gap opens, q. start, q. end, s. start, s. end, evalue, bit score\n# {} hits found",
                hits.len()
            )?;
        }
        for h in hits {
            let (qstart, qend, sstart, send) = h.coordinates();
            let (identical, mismatch, gapopen) = h.counts();
            let length = h.alignment.operations.len();
            let pident = 100.0 * identical as f64 / length as f64;
            if matches!(self.format, OutputFormat::Json) {
                if !self.first {
                    self.writer.write_all(b",\n")?;
                }
                self.first = false;
                // CIGAR is oriented along the original query, like the exported coordinates.
                let mut oriented = h.alignment.clone();
                if h.strand == Strand::Minus {
                    oriented.operations.reverse();
                }
                let value = serde_json::json!({"qseqid":h.query_id,"sseqid":h.subject_id,"pident":pident,"length":length,"mismatch":mismatch,"gapopen":gapopen,"qstart":qstart,"qend":qend,"sstart":sstart,"send":send,"evalue":h.evalue,"bitscore":h.bit_score,"score":h.alignment.summary.score,"strand":if h.strand==Strand::Plus {"plus"} else {"minus"},"cigar":oriented.cigar(),"band_edge_reached":h.alignment.summary.band_edge_reached});
                serde_json::to_writer(&mut self.writer, &value)
                    .map_err(|e| Error(e.to_string()))?;
            } else {
                let evalue = Evalue(h.evalue, self.ncbi_numbers);
                let bits = BitScore(h.bit_score, self.ncbi_numbers);
                writeln!(
                    self.writer,
                    "{}\t{}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    h.query_id,
                    h.subject_id,
                    pident,
                    length,
                    mismatch,
                    gapopen,
                    qstart,
                    qend,
                    sstart,
                    send,
                    evalue,
                    bits
                )?;
            }
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        if matches!(self.format, OutputFormat::Json) {
            self.writer.write_all(b"\n]\n")?;
        }
        self.writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ncbi_tabular_number_boundaries() {
        for (value, expected) in [
            (0.0, "0.0"),
            (1e-181, "0.0"),
            (1e-180, "1.00e-180"),
            (3.456e-9, "3.46e-09"),
            (0.0009, "0.001"),
            (0.03949, "0.039"),
            (0.1, "0.10"),
            (1.679, "1.7"),
            (10.0, "10"),
        ] {
            assert_eq!(ncbi_evalue(value), expected);
        }
        for (value, expected) in [
            (38.3, "38.3"),
            (99.9, "99.9"),
            (99.91, "99"),
            (1908.3, "1908"),
            (99999.0, "99999"),
            (100000.0, "1.000e+05"),
        ] {
            assert_eq!(ncbi_bits(value), expected);
        }
    }
}
