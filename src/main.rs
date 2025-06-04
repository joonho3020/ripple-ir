use clap::Parser;
use ripple_ir::common::read_annos;
use ripple_ir::common::write_annos;
use ripple_ir::passes::fir::to_ast_firrtl3::to_ast_firrtl3;
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
use ripple_ir::passes::ast::firrtl3_split_exprs::firrtl3_split_exprs;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input file path
    #[arg(short, long)]
    input: PathBuf,

    /// Output file path
    #[arg(short, long)]
    output: PathBuf,

    /// Input annotation path
    #[arg(long)]
    annos_in: Option<PathBuf>,

    /// Output file path
    #[arg(long)]
    annos_out: Option<PathBuf>,

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
            let mut circuit = parse_firrtl3(&source).expect("firrtl3 parser");
            let mut annos_opt = match args.annos_in {
                Some(annos_in) => {
                    Some(read_annos(annos_in.to_str().unwrap())?)
                }
                _ => None
            };


            firrtl3_split_exprs(&mut circuit, &mut annos_opt);
            let ir = run_fir_passes_from_circuit(&circuit)?;
            let circuit_reconstruct = to_ast_firrtl3(&ir);

            if let Some(annos) = annos_opt {
                assert!(args.annos_out.is_some(), "Annotations provided, but no output for annos");
                let annos_out = args.annos_out.unwrap();
                write_annos(&annos, annos_out.to_str().unwrap())?;
            }

            let mut printer = FIRRTL3Printer::new();
            printer.print_circuit(&circuit_reconstruct)
        }
    };

    std::fs::write(&args.output, circuit_str)?;

    Ok(())
}
