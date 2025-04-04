use chirrtl_parser::ast::Identifier;
use petgraph::Direction::{Incoming, Outgoing};
use petgraph::visit::EdgeRef;
use crate::ir::fir::FirGraph;
use crate::ir::fir::{FirEdgeType, FirIR, FirNodeType};
use crate::common::RippleIRErr;
use crate::common::graphviz::GraphViz;

/// Checks whether the phi nodes are connected properly to its parents and child
/// - This should run after remove_unnecessary_phi
/// - Each phi node should have at least one selection signal
/// - Each phi node should have exactly one child node
/// - Each phi node should have the same name as its child
pub fn check_phi_node_connections(fir: &FirIR) -> Result<(), RippleIRErr> {
    for (name, fgraph) in fir.graphs.iter() {
        check_phi_node_connections_graph(fgraph, name)?;
    }
    Ok(())
}

fn check_phi_node_connections_graph(fg: &FirGraph, name: &Identifier) -> Result<(), RippleIRErr> {
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
        if node.nt == FirNodeType::Phi {
            let mut child_cnt = 0;
            let childs = fg.graph.neighbors_directed(id, Outgoing);

            for cid in childs {
                child_cnt += 1;
                let child = fg.graph.node_weight(cid).unwrap();
                if child.name.as_ref().unwrap() != node.name.as_ref().unwrap() {
                    return Err(RippleIRErr::PhiNodeError(
                            format!("Phi node {:?} has a different name with its child {:?}",
                                node.name.as_ref().unwrap(), child.name.as_ref().unwrap())));
                }
            }

            if child_cnt != 1 {
                return Err(RippleIRErr::PhiNodeError(
                        format!("Phi node {:?} should have a single child, got {:?}",
                            node.name.as_ref().unwrap(), child_cnt)));
            }

            let mut phi_sel_cnt = 0;
            let mut phi_in_cnt = 0;
            let pedges = fg.graph.edges_directed(id, Incoming);
            for pedge in pedges {
                let ew = fg.graph.edge_weight(pedge.id()).unwrap();
                match &ew.et {
                    FirEdgeType::PhiSel => {
                        phi_sel_cnt += 1;
                    }
                    FirEdgeType::PhiInput(_pcond) => {
                        phi_in_cnt += 1;
                    }
                    _ => { }
                }
            }

            if phi_sel_cnt == 0 {
                fg.export_graphviz(&format!("./test-outputs/{:?}.removephi.pdf", name), None, None, false)?;
                return Err(RippleIRErr::PhiNodeError(
                        format!("Module {:?} Phi nodes {:?} should have at least one selection signal", name, node)))
            }

            if phi_in_cnt == 0 {
                return Err(RippleIRErr::PhiNodeError(
                        format!("Module {:?} Phi nodes {:?} should have at least one input signal", name, node)))
            }
        }
    }
    Ok(())
}
