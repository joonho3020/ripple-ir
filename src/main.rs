use clap::Parser;
use std::path::PathBuf;
use chirrtl_parser::parse_circuit;
use ripple_ir::passes::fir::from_ast::from_circuit;
use ripple_ir::passes::fir::remove_unnecessary_phi::remove_unnecessary_phi;
use ripple_ir::passes::fir::check_phi_nodes::check_phi_node_connections;
use ripple_ir::passes::fir::to_ast::to_ast;
use ripple_ir::passes::ast::print::Printer;
use ripple_ir::common::RippleIRErr;

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

    let mut ir = from_circuit(&circuit);
    remove_unnecessary_phi(&mut ir);
    check_phi_node_connections(&ir)?;

    let circuit_reconstruct = to_ast(&ir);

    let mut printer = Printer::new();
    let circuit_str = printer.print_circuit(&circuit_reconstruct);
    std::fs::write(&args.output, circuit_str)?;

    Ok(())
}
