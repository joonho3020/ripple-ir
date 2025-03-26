use crate::ir::firir::FirIR;
use crate::ir::RippleIR;
use crate::passes::from_ast::from_circuit;
use crate::passes::from_fir::from_fir;
use crate::passes::remove_unnecessary_phi::remove_unnecessary_phi;
use crate::common::RippleIRErr;
use crate::passes::infer_typetree::*;
use crate::timeit;
use chirrtl_parser::ast::Circuit;
use chirrtl_parser::parse_circuit;

use super::check_phi_nodes::check_phi_node_connections;

/// Run the AST to graph conversion and export the graph form
pub fn run_passes_from_filepath(name: &str) -> Result<FirIR, RippleIRErr> {
    let source = std::fs::read_to_string(name.to_string())?;

    let circuit = timeit!("firrtl parsing", {
        parse_circuit(&source)
    }).expect("firrtl parser");

    run_fir_passes(&circuit)
}

pub fn run_fir_passes(circuit: &Circuit) -> Result<FirIR, RippleIRErr> {
    let mut fir = from_circuit(&circuit);

    timeit!("remove_unnecessary_phi", {
        remove_unnecessary_phi(&mut fir);
        check_phi_node_connections(&fir)?;
    });

    timeit!("infer_typetree", {
        infer_typetree(&mut fir);
        check_typetree_inference(&fir)?;
    });

    Ok(fir)
}

pub fn run_rir_passes(fir: &FirIR) -> Result<RippleIR, RippleIRErr> {
    let rir = timeit!("from_fir", {
        from_fir(fir)
    });

    Ok(rir)
}
