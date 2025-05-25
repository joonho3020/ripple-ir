use clap::Parser;
use ripple_ir::passes::runner::run_fir_passes_from_circuit;
use std::path::PathBuf;
use chirrtl_parser::parse_circuit;
use ripple_ir::passes::fir::to_ast::to_ast;
use ripple_ir::passes::ast::chirrtl_print::ChirrtlPrinter;
use ripple_ir::common::RippleIRErr;
use ripple_ir::passes::ast::print::Printer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input file path
    #[arg(short, long)]
    input: PathBuf,

    /// Output file path
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> Result<(), RippleIRErr> {
    let args = Args::parse();

    let source = std::fs::read_to_string(args.input)?;
    let circuit = parse_circuit(&source).expect("firrtl parser");

    let ir = run_fir_passes_from_circuit(&circuit)?;
    let circuit_reconstruct = to_ast(&ir);

    let mut printer = ChirrtlPrinter::new();
    let circuit_str = printer.print_circuit(&circuit_reconstruct);
    std::fs::write(&args.output, circuit_str)?;

    Ok(())
}
