use blast_core::{Error, Result};
use blast_db::{DatabaseBuilder, FastaReader};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "makedb-rs",
    version,
    about = "Create a nucleotide RBLAST01 database (not an NCBI .nin/.nsq database)"
)]
struct Args {
    #[arg(long = "in")]
    input: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long,default_value="nucl",value_parser=["nucl"])]
    dbtype: String,
}
fn main() {
    blast_cli::run(|| {
        let args = Args::parse_from(blast_cli::arguments());
        if args.input == args.out {
            return Err(Error("input and database output must differ".into()));
        }
        let (count, bases) =
            DatabaseBuilder::build(FastaReader::new(blast_cli::input(&args.input)?), &args.out)?;
        eprintln!(
            "Created {}: {count} sequences, {bases} bases",
            args.out.display()
        );
        Ok(()) as Result<()>
    });
}
