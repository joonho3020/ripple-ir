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
use chirrtl_parser::parse_circuit;


/// Run the AST to graph conversion and export the graph form
pub fn run_passes_from_filepath(name: &str) -> Result<FirIR, RippleIRErr> {
    let source = std::fs::read_to_string(name.to_string())?;

    let circuit = timeit!("firrtl parsing", {
        parse_circuit(&source)
    }).expect("firrtl parser");

    run_fir_passes_from_circuit(&circuit)
}

pub fn run_fir_passes_from_circuit(circuit: &Circuit) -> Result<FirIR, RippleIRErr> {
    let mut fir = from_circuit(&circuit);
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
