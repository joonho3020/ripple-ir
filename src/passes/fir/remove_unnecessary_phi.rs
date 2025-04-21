use crate::ir::{fir::*, whentree::CondPathWithPrior};
use petgraph::{
    graph::{NodeIndex, EdgeIndex},
    visit::EdgeRef, Direction::{Incoming, Outgoing}
};
use chirrtl_parser::ast::*;
use indexmap::IndexMap;
use indexmap::IndexSet;

pub fn remove_unnecessary_phi(ir: &mut FirIR) {
    for (_id, rg) in ir.graphs.iter_mut() {
        remove_unnecessary_phi_in_ripple_graph(rg);
    }
}

fn remove_unnecessary_phi_in_ripple_graph(fg: &mut FirGraph) {
    let mut remove_nodes: Vec<NodeIndex> = vec![];
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
        match node.nt {
            FirNodeType::Phi(..) => {
                if is_removable(fg, id) {
                    connect_phi_parent_to_child(fg, id);
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
        fg.graph.remove_node(*id);
    }

    flip_bidirectional_edges_graph(fg);
}

/// Remove phi nodes when
/// - There is no selection signal
/// - The selection signal is always true
fn is_removable(fg: &FirGraph, id: NodeIndex) -> bool {
    let mut has_sel = false;
    let mut has_non_trivial_sel = false;

    let pedges = fg.graph.edges_directed(id, Incoming);
    for pedge in pedges {
        let edge = fg.graph.edge_weight(pedge.id()).unwrap();
        match &edge.et {
            FirEdgeType::PhiSel => {
                has_sel = true;
            }
            FirEdgeType::PhiInput(ppath, _flipped) => {
                if !has_non_trivial_sel {
                    has_non_trivial_sel = !ppath.path.always_true()
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
        let pedges = fg.graph.edges_directed(id, Incoming);
        for pedge in pedges {
            let edge = fg.graph.edge_weight(pedge.id()).unwrap();
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

fn connect_phi_parent_to_child(fg: &mut FirGraph, id: NodeIndex) {
    let childs: Vec<NodeIndex> = fg.graph.neighbors_directed(id, Outgoing).into_iter().collect();
    if childs.len() == 0 {
        return;
    }

    assert!(childs.len() == 1, "Phi node is driving multiple nodes {}", childs.len());

    let pedges: Vec<EdgeIndex> = fg.graph.edges_directed(id, Incoming).into_iter().map(|x| x.id()).collect();
    let mut dst_by_priority: IndexMap<Reference, (EdgeIndex, CondPathWithPrior)> = IndexMap::new();

    for peid in pedges.iter() {
        let ew = fg.graph.edge_weight(*peid).unwrap();
        let ep = fg.graph.edge_endpoints(*peid).unwrap();
        let src = ep.0;
        match &ew.et {
            FirEdgeType::PhiInput(ppath, _flipped) => {
                let dst_ref = ew.dst.as_ref().unwrap();
                if !dst_by_priority.contains_key(dst_ref) {
                    dst_by_priority.insert(dst_ref.clone(), (*peid, ppath.clone()));
                } else {
                    let cur_max = dst_by_priority.get(dst_ref).unwrap();
                    if &cur_max.1 > ppath {
                        dst_by_priority.insert(dst_ref.clone(), (*peid, ppath.clone()));
                    }
                }
            }
            FirEdgeType::DontCare => {
                fg.graph.add_edge(src, childs[0], ew.clone());
            }
            _ => {
                panic!("Phi node driver edge should be PhiInput, got {:?}", ew);
            }
        }
    }

    // For cases when there are multiple connects with the same priority connecting to the
    // same destination, just take the highest priority (for last connect semantics)
    for (_dst_ref, (eid, pconds)) in dst_by_priority {
        let ew = fg.graph.edge_weight(eid).unwrap();
        let src = fg.graph.edge_endpoints(eid).unwrap().0;
        let edge = FirEdge::new(ew.src.clone(), ew.dst.clone(), FirEdgeType::PhiInput(pconds.clone(), false));
        fg.graph.add_edge(src, childs[0], edge);
    }
}


#[derive(Debug)]
struct FlippedEdges {
    src: NodeIndex,
    dst: NodeIndex,
    edge: FirEdge,
}

type FlippedEdgeMap = IndexMap<EdgeIndex, FlippedEdges>;
type WiredEdgeSet = IndexSet<EdgeIndex>;

fn flip_dst_phi_node_removed_if_bidirectional(
    fg: &FirGraph,
    eid: EdgeIndex,
    flipped_edges: &mut FlippedEdgeMap,
    wire_edges: &mut WiredEdgeSet
    ) {
    if !fg.bidirectional(eid) {
        wire_edges.insert(eid);
    } else {
        let (src, dst) = fg.graph.edge_endpoints(eid).unwrap();
        let src_phi_parent_opt = fg.parent_with_type(src, FirEdgeType::PhiOut);

        if let Some(src_phi_eid) = src_phi_parent_opt {
            let (src_phi_id, _) = fg.graph.edge_endpoints(src_phi_eid).unwrap();

            let edge = fg.graph.edge_weight(eid).unwrap();
            match (&edge.src, &edge.dst) {
                (Expr::Reference(src_ref), Some(dst_ref)) => {
                    let flipped_src = Expr::Reference(dst_ref.clone());
                    let flipped_dst = Some(src_ref.clone());
                    if let FirEdgeType::PhiInput(pconds, _) = &edge.et {
                        let et_flipped = FirEdgeType::PhiInput(pconds.clone(), true);
                        let flipped = FirEdge::new(flipped_src, flipped_dst, et_flipped);
                        flipped_edges.insert(eid, FlippedEdges { src: dst, dst: src_phi_id, edge: flipped });
                    }
                }
                _ => {
                    panic!("Bidirectional edge {:?} has no Reference as src or some dst", edge);
                }
            }
        } else {
            wire_edges.insert(eid);
        }
    }
}

/// There were cases where the phi node of the dst edge has been removed
/// However, the source node still has a phi node
/// The bidirectional src to dst connection is inserted in the wrong place in the whentree as the phi node of the dst is removed and we no longer know the exact priority
/// Solved this by searching for these cases and flipping the connection so that the edge heads towards the phi node
fn flip_bidirectional_edges_graph(fg: &mut FirGraph) {
    let mut flipped_edges: FlippedEdgeMap = FlippedEdgeMap::new();
    let mut wire_edges: WiredEdgeSet = WiredEdgeSet::new();

    for eid in fg.graph.edge_indices() {
        let (_src_id, dst_id) = fg.graph.edge_endpoints(eid).unwrap();
        let edge = fg.graph.edge_weight(eid).unwrap();
        match &edge.et {
            FirEdgeType::PhiInput(..) => {
                let dst = fg.graph.node_weight(dst_id).unwrap();
                if !dst.is_phi() {
                    flip_dst_phi_node_removed_if_bidirectional(fg, eid, &mut flipped_edges, &mut wire_edges);
                }
            }
            _ => {}
        }
    }


    for eid in fg.graph.edge_indices().rev() {
        if flipped_edges.contains_key(&eid) {
            fg.graph.remove_edge(eid);

            let flipped = flipped_edges.get(&eid).unwrap();
            fg.graph.add_edge(flipped.src, flipped.dst, flipped.edge.clone());
        } else if wire_edges.contains(&eid) {
            let edge = fg.graph.edge_weight_mut(eid).unwrap();
            edge.et = FirEdgeType::Wire;
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
    use crate::passes::fir::infer_typetree::*;
    use chirrtl_parser::parse_circuit;

    /// Run the AST to graph conversion and export the graph form
    fn run(name: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(format!("./test-inputs/{}.fir", name))?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let mut ir = from_circuit(&circuit);

        infer_typetree(&mut ir);
        check_typetree_inference(&ir)?;

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
