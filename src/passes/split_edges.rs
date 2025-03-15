use petgraph::visit::EdgeRef;
use petgraph::Direction::{Incoming, Outgoing};
use chirrtl_parser::ast::*;

use crate::ir::firir::*;

pub fn split_edges(ir: &mut FirIR) {
}

pub fn infer_typetree(ir: &mut FirIR) {
    for (_, rg) in ir.graphs.iter_mut() {
        infer_typetree_graph(rg);
    }
}

fn infer_typetree_graph(rg: &mut FirGraph) {
    for id in rg.graph.node_indices() {
        let node = rg.graph.node_weight(id).unwrap();
        let pedges = rg.graph.edges_directed(id, Incoming);
        let cedges = rg.graph.edges_directed(id, Outgoing);

        match &node.nt {
            FirNodeType::Invalid             |
                FirNodeType::DontCare        |
                FirNodeType::UIntLiteral(..) |
                FirNodeType::SIntLiteral(..) => {
                // Don't need to define typetree
            }
            FirNodeType::Mux => {
                let mut true_expr_opt:  Option<&Expr> = None;
                let mut false_expr_opt: Option<&Expr> = None;
                let mut cond_expr_opt: Option<&Expr> = None;
                for peid in pedges {
                    let pedge = rg.graph.edge_weight(peid.id()).unwrap();
                    match pedge.et {
                        FirEdgeType::MuxTrue => {
                            true_expr_opt = Some(&pedge.src);
                        }
                        FirEdgeType::MuxFalse => {
                            false_expr_opt = Some(&pedge.src);
                        }
                        FirEdgeType::MuxCond => {
                            cond_expr_opt = Some(&pedge.src);
                        }
                        _ => {
                            panic!("Unrecognized input edge {:?} for Mux", pedge);
                        }
                    }
                }
                match (true_expr_opt.unwrap(), false_expr_opt.unwrap()) {
                    (Expr::Reference(tref), Expr::Reference(fref)) => {

                    }
                    (Expr::UIntInit(..), _)    |
                    (Expr::SIntInit(..), _)    |
                    (Expr::PrimOp1Expr(..), _) |
                    (_, Expr::UIntInit(..))    |
                    (_, Expr::SIntInit(..))    |
                    (_, Expr::PrimOp1Expr(..)) => {
                        // These are ground types
                    }
                    _ => {
                        panic!("Unexpected edges going into Mux t: {:?} f: {:?}",
                            true_expr_opt, false_expr_opt);
                    }
                }

                todo!("Handle mux");
            }
            FirNodeType::PrimOp2Expr(..) => {
                todo!("Handle primop 2 expr");
            }
            FirNodeType::PrimOp1Expr(..) |
                FirNodeType::PrimOp1Expr1Int(..) |
                FirNodeType::PrimOp1Expr2Int(..) => {
                todo!("Handle primop 1 expr");
            }
            FirNodeType::Wire          |
                FirNodeType::Reg       |
                FirNodeType::RegReset  |
                FirNodeType::SMem(..)  |
                FirNodeType::CMem => {
                assert!(node.ttree.is_some());
            }
            FirNodeType::WriteMemPort     |
                FirNodeType::ReadMemPort  |
                FirNodeType::InferMemPort => {
                todo!("Handle memports");
            }
            FirNodeType::Inst(module) => {
                todo!("Handle inst");
            }
            FirNodeType::Input      |
                FirNodeType::Output => {
                assert!(node.ttree.is_some());
            }
            FirNodeType::Phi => {
                todo!("Handle phi");
            }
        }
    }
}
