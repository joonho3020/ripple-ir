use chirrtl_parser::ast::Expr;
use indexmap::IndexMap;
use petgraph::graph::NodeIndex;
use petgraph::graph::EdgeIndex;
use crate::{common::graphviz::DefaultGraphVizCore, ir::fir::{FirGraph, FirIR}};
use crate::ir::fir::*;

pub fn flip_bidirectional_edges(fir: &mut FirIR) {
    for (_name, fg) in fir.graphs.iter_mut() {
        flip_bidirectional_edges_graph(fg);
    }
}

#[derive(Debug)]
struct FlippedEdges {
    src: NodeIndex,
    dst: NodeIndex,
    edge: FirEdge,
}

fn flip_bidirectional_edges_graph(fg: &mut FirGraph) {
    let mut flipped_edges: IndexMap<EdgeIndex, FlippedEdges> = IndexMap::new();

    for eid in fg.edge_indices() {
        if !fg.bidirectional(eid) {
            continue;
        }
        let (src, dst) = fg.graph.edge_endpoints(eid).unwrap();
        let src_phi_parent_opt = fg.parent_with_type(src, FirEdgeType::PhiOut);
        let dst_node = fg.graph.node_weight(dst).unwrap();

        // This edge is a bidirectional edge
        // Src has a phi node
        if let Some(src_phi_eid) = src_phi_parent_opt {
            let (src_phi_id, _) = fg.graph.edge_endpoints(src_phi_eid).unwrap();

            // Dst doesn't have a phi node
            if !dst_node.is_phi() {
                let edge = fg.graph.edge_weight(eid).unwrap();

                match (&edge.src, &edge.dst) {
                    (Expr::Reference(src_ref), Some(dst_ref)) => {
                        let flipped_src = Expr::Reference(dst_ref.clone());
                        let flipped_dst = Some(src_ref.clone());
                        let flipped = FirEdge::new(flipped_src, flipped_dst, edge.et.clone());
                        flipped_edges.insert(eid, FlippedEdges { src: dst, dst: src_phi_id, edge: flipped });
                    }
                    _ => {}
                }
            }
        }
    }
    println!("flipped edges {:?}", flipped_edges);

    for eid in fg.edge_indices().rev() {
        if flipped_edges.contains_key(&eid) {
            fg.graph.remove_edge(eid);

            let flipped = flipped_edges.get(&eid).unwrap();
            fg.graph.add_edge(flipped.src, flipped.dst, flipped.edge.clone());
        }
    }
}
