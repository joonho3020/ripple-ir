use crate::ir::fir::FirIR;
use crate::ir::rir::rir::RippleIR;
use crate::passes::fir::from_ast::from_circuit;
use crate::passes::fir::infer_typetree::*;
use crate::passes::fir::remove_unnecessary_phi::remove_unnecessary_phi;
use crate::passes::fir::check_phi_nodes::check_phi_node_connections;
use crate::passes::rir::from_fir::from_fir;
use crate::common::RippleIRErr;
use crate::timeit;
use rusty_firrtl::Circuit;
use chirrtl_parser::parse_circuit as parse_chirrtl;
use firrtl3_parser::parse_circuit as parse_firrtl3;
use super::ast::check_ast_assumption::check_ast_assumption;
use super::ast::firrtl3_split_exprs::firrtl3_split_exprs;


/// Run the AST to graph conversion and export the graph form
pub fn run_passes_from_chirrtl_file(name: &str) -> Result<FirIR, RippleIRErr> {
    let source = std::fs::read_to_string(name.to_string())?;

    let circuit = timeit!("chirrtl parsing", {
        parse_chirrtl(&source)
    }).expect("firrtl parser");

    run_fir_passes_from_circuit(&circuit)
}

/// Run the AST to graph conversion and export the graph form
pub fn run_passes_from_firrtl3_file(name: &str) -> Result<FirIR, RippleIRErr> {
    let source = std::fs::read_to_string(name.to_string())?;

    let mut circuit = timeit!("firrtl3 parsing", {
        parse_firrtl3(&source)
    }).expect("firrtl parser");

    firrtl3_split_exprs(&mut circuit, &mut None);

use crate::passes::ast::firrtl3_print::FIRRTL3Printer;
use crate::passes::ast::print::*;
let mut printer = FIRRTL3Printer::new();
let reconstructed_circuit_str = printer.print_circuit(&circuit);
let out_path = format!("./test-outputs/{}.firrtl3.split.fir", circuit.name);
std::fs::write(&out_path, reconstructed_circuit_str)?;

    run_fir_passes_from_circuit(&circuit)
}


pub fn run_fir_passes_from_circuit(circuit: &Circuit) -> Result<FirIR, RippleIRErr> {
    check_ast_assumption(circuit);
    let mut fir = from_circuit(&circuit);
// fir.export("./test-outputs", "firrtl3")?;
    run_fir_passes(&mut fir)?;
    Ok(fir)
}

pub fn run_fir_passes(fir: &mut FirIR) -> Result<(), RippleIRErr> {
    timeit!("infer_typetree", {
        infer_typetree(fir);
        check_typetree_inference(&fir)?;
    });

    timeit!("remove_unnecessary_phi", {
        remove_unnecessary_phi(fir);
        check_phi_node_connections(&fir)?;
    });
    Ok(())
}

pub fn run_rir_passes(fir: &FirIR) -> Result<RippleIR, RippleIRErr> {
    let rir = timeit!("from_fir", {
        from_fir(fir)
    });

    Ok(rir)
}
