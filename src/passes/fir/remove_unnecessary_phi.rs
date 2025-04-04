use crate::ir::fir::*;
use petgraph::{
    graph::{NodeIndex, EdgeIndex},
    visit::EdgeRef, Direction::{Incoming, Outgoing}
};

pub fn remove_unnecessary_phi(ir: &mut FirIR) {
    for (_id, rg) in ir.graphs.iter_mut() {
        remove_unnecessary_phi_in_ripple_graph(rg);
    }
}

fn remove_unnecessary_phi_in_ripple_graph(rg: &mut FirGraph) {
    let mut remove_nodes: Vec<NodeIndex> = vec![];
    for id in rg.graph.node_indices() {
        let node = rg.graph.node_weight(id).unwrap();
        match node.nt {
            FirNodeType::Phi => {
                if is_removable(rg, id) {
                    connect_phi_parent_to_child(rg, id);
                    remove_nodes.push(id);
                }
            }
            _ => {
                continue;
            }
        }
    }

    remove_nodes.sort();

    for id in remove_nodes.iter().rev() {
        rg.graph.remove_node(*id);
    }
}

/// Remove phi nodes when
/// - There is no selection signal
/// - The selection signal is always true
fn is_removable(rg: &FirGraph, id: NodeIndex) -> bool {
    let mut has_sel = false;
    let mut has_non_trivial_sel = false;

    let pedges = rg.graph.edges_directed(id, Incoming);
    for pedge in pedges {
        let edge = rg.graph.edge_weight(pedge.id()).unwrap();
        match &edge.et {
            FirEdgeType::PhiSel => {
                has_sel = true;
            }
            FirEdgeType::PhiInput(pcond) => {
                if !has_non_trivial_sel {
                    has_non_trivial_sel = !pcond.conds.always_true()
                }
            }
            FirEdgeType::DontCare => {
            }
            _ => {
                panic!("Unrecognized driving edge {:?} for phi node", edge);
            }
        }
    }

    if has_sel && !has_non_trivial_sel {
        let pedges = rg.graph.edges_directed(id, Incoming);
        for pedge in pedges {
            let edge = rg.graph.edge_weight(pedge.id()).unwrap();
            eprintln!("{:?}", edge);
        }
        panic!("Phi node has incoming sel, but only has trivial selectors");
    }

    if !has_sel || !has_non_trivial_sel {
        return true;
    } else {
        return false;
    }
}

fn connect_phi_parent_to_child(rg: &mut FirGraph, id: NodeIndex) {
    let childs: Vec<NodeIndex> = rg.graph.neighbors_directed(id, Outgoing).into_iter().collect();
    if childs.len() == 0 {
        return;
    }

    assert!(childs.len() == 1, "Phi node is driving multiple nodes {}", childs.len());

    let pedges: Vec<EdgeIndex> = rg.graph.edges_directed(id, Incoming).into_iter().map(|x| x.id()).collect();
    for peid in pedges.iter() {
        let ew = rg.graph.edge_weight(*peid).unwrap();
        let ep = rg.graph.edge_endpoints(*peid).unwrap();
        let src = ep.0;
        match &ew.et {
            FirEdgeType::PhiInput(_pcond) => {
                let edge = FirEdge::new(ew.src.clone(), ew.dst.clone(), FirEdgeType::Wire);
                rg.graph.add_edge(src, childs[0], edge);
            }
            FirEdgeType::DontCare => {
                rg.graph.add_edge(src, childs[0], ew.clone());
            }
            _ => {
                panic!("Phi node driver edge should be PhiInput, got {:?}", ew);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        common::RippleIRErr,
        passes::fir::from_ast::from_circuit,
        passes::fir::check_phi_nodes::check_phi_node_connections,
    };
    use chirrtl_parser::parse_circuit;

    /// Run the AST to graph conversion and export the graph form
    fn run(name: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(format!("./test-inputs/{}.fir", name))?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let mut ir = from_circuit(&circuit);
        remove_unnecessary_phi(&mut ir);
        check_phi_node_connections(&ir)?;
        Ok(())
    }

    #[test]
    fn gcd() {
        run("GCD").expect("GCD");
    }

    #[test]
    fn nestedwhen() {
        run("NestedWhen").expect("NestedWhen");
    }

    #[test]
    fn nestedbundle() {
        run("NestedBundle").expect("NestedBundle");
    }

    #[test]
    fn singleport_sram() {
        run("SinglePortSRAM").expect("SinglePortSRAM");
    }

    #[test]
    fn hierarchy() {
        run("Hierarchy").expect("Hierarchy");
    }

    #[test]
    fn rocket() {
        run("chipyard.harness.TestHarness.RocketConfig").expect("rocket failed");
    }

    #[test]
    fn boom() {
        run("chipyard.harness.TestHarness.LargeBoomV3Config").expect("boom failed");
    }
}
