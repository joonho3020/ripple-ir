use clap::Parser;
use std::path::PathBuf;
use chirrtl_parser::parse_circuit as parse_chirrtl;
use firrtl3_parser::parse_circuit as parse_firrtl3;
use ripple_ir::common::RippleIRErr;
use ripple_ir::common::FIRRTLVersion;
use ripple_ir::passes::ast::print::Printer;
use ripple_ir::passes::ast::chirrtl_print::ChirrtlPrinter;
use ripple_ir::passes::ast::firrtl3_print::FIRRTL3Printer;
use ripple_ir::passes::fir::to_ast::to_ast;
use ripple_ir::passes::runner::run_fir_passes_from_circuit;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input file path
    #[arg(short, long)]
    input: PathBuf,

    /// Output file path
    #[arg(short, long)]
    output: PathBuf,

    /// Version of the FIRRTL input file
    #[arg(long, value_enum)]
    firrtl_version: FIRRTLVersion,
}

fn main() -> Result<(), RippleIRErr> {
    let args = Args::parse();

    let source = std::fs::read_to_string(args.input)?;
    let circuit_str = match args.firrtl_version {
        FIRRTLVersion::Chirrtl => {
            let circuit = parse_chirrtl(&source).expect("chirrtl parser");
            let ir = run_fir_passes_from_circuit(&circuit)?;
            let circuit_reconstruct = to_ast(&ir);

            let mut printer = ChirrtlPrinter::new();
            printer.print_circuit(&circuit_reconstruct)
        }
        FIRRTLVersion::Firrtl3 => {
            let circuit = parse_firrtl3(&source).expect("firrtl3 parser");
            let mut printer = FIRRTL3Printer::new();
            printer.print_circuit(&circuit)
        }
    };

    std::fs::write(&args.output, circuit_str)?;

    Ok(())
}
