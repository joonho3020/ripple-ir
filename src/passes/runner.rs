use crate::ir::firir::FirIR;
use crate::passes::from_ast::from_circuit;
use crate::passes::remove_unnecessary_phi::remove_unnecessary_phi;
use crate::common::RippleIRErr;
use crate::passes::infer_typetree::infer_typetree;
use chirrtl_parser::ast::Circuit;
use chirrtl_parser::parse_circuit;

/// Run the AST to graph conversion and export the graph form
pub fn run_passes_from_filepath(name: &str) -> Result<FirIR, RippleIRErr> {
    let source = std::fs::read_to_string(name.to_string())?;
    let circuit = parse_circuit(&source).expect("firrtl parser");
    Ok(run(&circuit))
}

pub fn run(circuit: &Circuit) -> FirIR {
    let mut ir = from_circuit(&circuit);
    remove_unnecessary_phi(&mut ir);
    infer_typetree(&mut ir);
    return ir;
}
