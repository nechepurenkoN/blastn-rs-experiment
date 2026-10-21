//! Nucleotide seed-and-extend search. Internal coordinates are zero-based, half-open.
pub mod align;
pub mod dna;
pub mod mask;
pub mod search;
pub mod seed;
pub mod stats;

use std::fmt;

#[derive(Debug, Clone)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}
pub type Result<T> = std::result::Result<T, Error>;
