use chirrtl_parser::ast::*;
use image::imageops::colorops::contrast_in_place;
use petgraph::{visit::EdgeRef, Direction::Incoming};

use crate::{common::graphviz::DefaultGraphVizCore, ir::{fir::{FirEdgeType, FirGraph, FirIR, FirNodeType}, whentree::{PrioritizedCond, WhenTree}}};

pub fn to_ast(fir: &FirIR) -> Option<Circuit> {
    for (name, fgraph) in fir.graphs.iter() {
    }
    None
}

fn to_circuitmodule(name: &Identifier, fg: &FirGraph) -> CircuitModule {
    if fg.blackbox {
        CircuitModule::ExtModule(to_extmodule(name, fg))
    } else {
        CircuitModule::Module(to_module(name, fg))
    }
}

fn to_extmodule(name: &Identifier, fg: &FirGraph) -> ExtModule {
    let ext_info = &fg.ext_info.as_ref().unwrap();
    let defname = &ext_info.defname;
    let params = &ext_info.params;
    let ports = get_ports(fg);
    ExtModule::new(name.clone(), ports, defname.clone(), params.clone(), Info::default())
}

fn get_ports(fg: &FirGraph) -> Ports {
    let mut ret = Ports::new();
    for id in fg.node_indices() {
        let node = fg.node_weight(id).unwrap();
        match node.nt {
            FirNodeType::Input => {
                ret.push(Box::new(Port::Input(
                    node.name.as_ref().unwrap().clone(),
                    node.ttree.as_ref().unwrap().to_type(),
                    Info::default())));
            }
            FirNodeType::Output => {
                ret.push(Box::new(Port::Output(
                    node.name.as_ref().unwrap().clone(),
                    node.ttree.as_ref().unwrap().to_type(),
                    Info::default())));
            }
            _ => {
                continue;
            }
        }
    }
    return ret;
}


fn to_module(name: &Identifier, fg: &FirGraph) -> Module {
    let ports = get_ports(fg);
    let mut stmts: Stmts = Stmts::new();
    collect_def_stmts(fg, &mut stmts);
    Module::new(name.clone(), ports, stmts, Info::default())
}

/// Statements that defines a structural element
fn collect_def_stmts(fg: &FirGraph, stmts: &mut Stmts) {
    for id in fg.node_indices() {
        let node = fg.node_weight(id).unwrap();
        let tpe = node.ttree.as_ref().unwrap().to_type();
        let name = node.name.as_ref().unwrap().clone();

        match &node.nt {
            FirNodeType::Wire => {
                stmts.push(Box::new(Stmt::Wire(name, tpe, Info::default())));
            }
            FirNodeType::Reg => {
                let clk_eid = fg.parent_with_type(id, FirEdgeType::Clock).unwrap();
                let clk = fg.graph.edge_weight(clk_eid).unwrap().src.clone();

                let reg = Stmt::Reg(name, tpe, clk, Info::default());
                stmts.push(Box::new(reg));
            }
            FirNodeType::RegReset => {
                let clk_eid = fg.parent_with_type(id, FirEdgeType::Clock).unwrap();
                let rst_eid = fg.parent_with_type(id, FirEdgeType::Reset).unwrap();
                let init_eid = fg.parent_with_type(id, FirEdgeType::Wire).unwrap();

                let clk  = fg.graph.edge_weight(clk_eid).unwrap().src.clone();
                let rst  = fg.graph.edge_weight(rst_eid).unwrap().src.clone();
                let init = fg.graph.edge_weight(init_eid).unwrap().src.clone();

                let reg_init = Stmt::RegReset(name, tpe, clk, rst, init, Info::default());
                stmts.push(Box::new(reg_init));
            }
            FirNodeType::SMem(ruw_opt) => {
                let smem = ChirrtlMemory::SMem(name, tpe, ruw_opt.clone(), Info::default());
                let mem = Stmt::ChirrtlMemory(smem);
                stmts.push(Box::new(mem));
            }
            FirNodeType::CMem => {
                let cmem = ChirrtlMemory::CMem(name, tpe, Info::default());
                let mem = Stmt::ChirrtlMemory(cmem);
                stmts.push(Box::new(mem));
            }
            FirNodeType::Inst(module) => {
                let inst = Stmt::Inst(name, module.clone(), Info::default());
                stmts.push(Box::new(inst));
            }
            _ => {
                continue;
            }
        }
    }
}

fn reconstruct_whentree(fg: &FirGraph) -> WhenTree {
    let mut cond_priority_pair = vec![];
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
        match &node.nt {
            FirNodeType::Phi => {
                let eids = fg.graph.edges_directed(id, Incoming);
                for eid in eids {
                    let edge = fg.graph.edge_weight(eid.id()).unwrap();
                    match &edge.et {
                        FirEdgeType::PhiInput(prior, cond) => {
                            println!("cond {:?}", cond);
                            cond_priority_pair.push(PrioritizedCond::new(prior.clone(), cond.clone()));
                        }
                        _ => {
                        }
                    }
                }

            }
            _ => {
                continue;
            }
        }
    }

    WhenTree::from_conditions(cond_priority_pair)


}

fn collect_remaining_stmts(fg: &FirGraph, stmts: &mut Stmts) {
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::passes::fir::from_ast::from_circuit;
    use crate::passes::fir::remove_unnecessary_phi::remove_unnecessary_phi;
    use crate::passes::fir::check_phi_nodes::check_phi_node_connections;
    use crate::common::RippleIRErr;
    use chirrtl_parser::parse_circuit;

    fn run(name: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(format!("./test-inputs/{}.fir", name))?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let mut ir = from_circuit(&circuit);
        remove_unnecessary_phi(&mut ir);
        check_phi_node_connections(&ir)?;

        for (_name, fg) in ir.graphs.iter() {
            let whentree = reconstruct_whentree(fg);
            whentree.print_tree();
        }
        Ok(())
    }

    #[test]
    fn gcd() -> Result<(), RippleIRErr> {
        run("GCD")?;
        Ok(())
    }

    #[test]
    fn nested_when() -> Result<(), RippleIRErr> {
        run("NestedWhen")?;
        Ok(())
    }
}
