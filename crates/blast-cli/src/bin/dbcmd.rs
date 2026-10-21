use blast_core::Error;
use blast_db::{BlastDatabase, SequenceStore, write_fasta};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "dbcmd-rs",
    version,
    about = "Extract FASTA or metadata from an RBLAST01 database"
)]
struct Args {
    #[arg(long)]
    db: PathBuf,
    #[arg(long, default_value = "all")]
    entry: String,
    #[arg(long,default_value="fasta",value_parser=["fasta","metadata"])]
    outfmt: String,
    #[arg(long)]
    out: Option<PathBuf>,
}
fn main() {
    blast_cli::run(|| {
        let args = Args::parse_from(blast_cli::arguments());
        if args.out.as_ref() == Some(&args.db) {
            return Err(Error("output must differ from database".into()));
        }
        let db = BlastDatabase::open(&args.db)?;
        let mut out = blast_cli::output(args.out.as_deref())?;
        let mut found = false;
        for i in 0..db.len() {
            let r = db.record(i)?;
            if args.entry != "all" && args.entry != r.id {
                continue;
            }
            found = true;
            if args.outfmt == "metadata" {
                writeln!(out, "{}\t{}\t{}", r.id, r.sequence.len(), r.description)?;
            } else {
                write_fasta(&r, &mut out)?;
            }
        }
        if !found {
            return Err(Error(format!("unknown database entry: {}", args.entry)));
        }
        out.flush()?;
        Ok(())
    });
}
