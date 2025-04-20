use chirrtl_parser::ast::Expr;

use crate::{common::{graphviz::DefaultGraphVizCore, RippleIRErr}, ir::fir::{FirGraph, FirIR}};
use crate::ir::fir::*;

pub fn add_bidirectional_edges(fir: &mut FirIR) {
    for (_name, fg) in fir.graphs.iter_mut() {
        add_bidirectional_edges_graph(fg);
    }
}

fn add_bidirectional_edges_graph(fg: &mut FirGraph) {
    for eid in fg.edge_indices() {
        let edge = fg.graph.edge_weight(eid).unwrap();
        let (src, dst) = fg.graph.edge_endpoints(eid).unwrap();

        let ttree = fg.graph.node_weight(src).unwrap()
            .ttree.as_ref().unwrap()
            .view().unwrap();

        let subtree = ttree.subtree_from_expr(&edge.src).unwrap();
        if !subtree.is_bidirectional() {
            continue;
        }

        let src_node = fg.graph.node_weight(src).unwrap();
        let dst_node = fg.graph.node_weight(dst).unwrap();

        let src_in_id = if src_node.is_phi() {
            let src_phi_id = fg.parent_with_type(src, FirEdgeType::PhiOut).unwrap();
            let ep = fg.graph.edge_endpoints(src_phi_id).unwrap();
            ep.0
        } else {
            src
        };

        let dst_out_id = if dst_node.is_phi() {
            let childs = fg.childs_with_type(dst, FirEdgeType::PhiOut);
            assert!(childs.len() == 1);
            fg.graph.edge_endpoints(childs[0]).unwrap().1
        } else {
            dst
        };

        if edge.dst.is_some() && edge.et != FirEdgeType::DontCare {
            let src_expr = Expr::Reference(edge.dst.as_ref().unwrap().clone());
            if let Expr::Reference(dst_ref) = &edge.src {
                let reverse_edge = FirEdge::new(src_expr, Some(dst_ref.clone()), edge.et.clone());
                fg.graph.add_edge(dst_out_id, src_in_id, reverse_edge);
            }
        }
    }
}

pub fn check_bidirectional_edges(fir: &FirIR) -> Result<(), RippleIRErr> {
    Ok(())
}
