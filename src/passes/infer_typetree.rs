use std::collections::VecDeque;
use chirrtl_parser::ast::PrimOp1Expr;
use petgraph::Direction::{Incoming, Outgoing};
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeRef, VisitMap, Visitable};
use petgraph::Undirected;
use petgraph::prelude::Dfs;
use indexmap::IndexMap;
use crate::ir::firir::*;
use crate::ir::typetree::*;

pub fn infer_typetree(ir: &mut FirIR) {
    for (_, rg) in ir.graphs.iter_mut() {
        infer_typetree_graph(rg);
    }
}

fn set_ground_type(rg: &mut FirGraph, id: NodeIndex, gt: GroundType) {
    let node_mut = rg.graph.node_weight_mut(id).unwrap();
    node_mut.ttree = Some(TypeTree::build_from_ground_type(gt));
}

fn topo_start_node(nt: &FirNodeType) -> bool {
    match nt {
        FirNodeType::Invalid             |
            FirNodeType::DontCare        |
            FirNodeType::UIntLiteral(..) |
            FirNodeType::SIntLiteral(..) |
            FirNodeType::Wire            |
            FirNodeType::Reg             |
            FirNodeType::RegReset        |
            FirNodeType::SMem(..)        |
            FirNodeType::CMem            |
            FirNodeType::ReadMemPort     |
            FirNodeType::WriteMemPort    |
            FirNodeType::InferMemPort    |
            FirNodeType::Inst(..)        |
            FirNodeType::Input           |
            FirNodeType::Output          |
            FirNodeType::Phi => {
                return true;
            }
        _ => {
            return false;
        }
    }
}

fn infer_typetree_graph(rg: &mut FirGraph) {
    use crate::ir::typetree::GroundType::*;

    // Take care of nodes with ground types and memory
    for id in rg.graph.node_indices() {
        let nt = &rg.graph.node_weight(id).unwrap().nt;
        match nt {
            FirNodeType::Invalid         => { set_ground_type(rg, id, Invalid); }
            FirNodeType::DontCare        => { set_ground_type(rg, id, DontCare); }
            FirNodeType::UIntLiteral(..) => { set_ground_type(rg, id, UInt); }
            FirNodeType::SIntLiteral(..) => { set_ground_type(rg, id, SInt); }
            FirNodeType::Inst(..)        => { set_ground_type(rg, id, Inst); }
            FirNodeType::CMem |
                FirNodeType::SMem(..) => {
                let ttree = rg.graph.node_weight(id).unwrap().ttree.clone();

                // Copy ttree to each port
                let childs: Vec<NodeIndex> = rg.graph.neighbors_directed(id, Outgoing).collect();
                for cid in childs {
                    let child = rg.graph.node_weight_mut(cid).unwrap();
                    match child.nt {
                        FirNodeType::InferMemPort |
                            FirNodeType::WriteMemPort |
                            FirNodeType::ReadMemPort => {
                            child.ttree = Some(ttree.as_ref().unwrap().clone());
                        }
                        _ => {
                            panic!("Unexpected child node {:?} for memory", child);
                        }
                    }
                }
            }
            FirNodeType::Phi => {
                let childs: Vec<NodeIndex> = rg.graph.neighbors_directed(id, Outgoing).collect();
                assert!(childs.len() == 1, "Phi node should have a single child");

                let cid = childs[0];
                let child_ttree = rg.graph.node_weight(cid).unwrap().ttree.clone();
                let node_mut = rg.graph.node_weight_mut(id).unwrap();
                node_mut.ttree = child_ttree;

                assert!(node_mut.ttree.is_some());
            }
            _ => {
            }
        }
    }

    // compute indeg for the entire graph
    let mut indeg: IndexMap<NodeIndex, u32> = IndexMap::new();
    for id in rg.graph.node_indices() {
        indeg.insert(id, 0);
    }
    for eid in rg.graph.edge_indices() {
        let ep = rg.graph.edge_endpoints(eid).unwrap();
        let dst = ep.1;
        *indeg.get_mut(&dst).unwrap() += 1;
    }

    let undir_graph = rg.graph.clone().into_edge_type::<Undirected>();
    let mut visited = 0;
    let mut vis_map = rg.graph.visit_map();
    for id in rg.graph.node_indices() {
        if vis_map.is_visited(&id) {
            continue;
        }

        let mut q: VecDeque<NodeIndex> = VecDeque::new();
        let mut dfs = Dfs::new(&undir_graph, id);
        while let Some(nx) = dfs.next(&undir_graph) {
            vis_map.visit(nx);

            let node = rg.graph.node_weight(nx).unwrap();
            if topo_start_node(&node.nt) {
                q.push_back(nx);
            }
        }

        // Start topological sort
        let mut topo_sort_order: Vec<NodeIndex> = vec![];
        let mut topo_vis_map = rg.graph.visit_map();
        while !q.is_empty() {
            let nidx = q.pop_front().unwrap();
            if topo_vis_map.is_visited(&nidx) {
                continue;
            }

            topo_vis_map.visit(nidx);
            topo_sort_order.push(nidx);

            let childs = rg.graph.neighbors_directed(nidx, Outgoing);
            for cidx in childs {
                let child = rg.graph.node_weight(cidx).unwrap();
                if !topo_vis_map.is_visited(&cidx) && !topo_start_node(&child.nt) {
                    *indeg.get_mut(&cidx).unwrap() -= 1;
                    if *indeg.get(&cidx).unwrap() == 0 {
                        q.push_back(cidx);
                    }
                }
            }
        }

        // Infer in topo sorted order
        for nidx in topo_sort_order.iter() {
            let node = rg.graph.node_weight(id).unwrap();
            if node.ttree.is_some() {
                continue;
            }
            infer_typetree_node(rg, *nidx);
        }

        visited += topo_sort_order.len();
    }
    assert!(
        visited == vis_map.len(),
        "visited {} nodes out of {} nodes while topo sorting",
        visited,
        vis_map.len());
}

fn infer_typetree_node(rg: &mut FirGraph, id: NodeIndex) {
    let node = rg.graph.node_weight(id).unwrap();
    assert!(!node.ttree.is_some());

    type EdgeEndpoint = (NodeIndex, NodeIndex);
    type EdgeWeightEpTuple<'a> = (&'a FirEdge, EdgeEndpoint);

    match node.nt {
        FirNodeType::Mux => {
            let incoming = rg.graph.edges_directed(id, Incoming)
                .map(|x|
                    (
                        rg.graph.edge_weight(x.id()).unwrap(),
                        rg.graph.edge_endpoints(x.id()).unwrap()
                    )
                );
            let true_edge_vec: Vec<EdgeWeightEpTuple> = incoming.clone()
                .filter(|x| x.0.et == FirEdgeType::MuxTrue).collect();

            let false_edge_vec: Vec<EdgeWeightEpTuple> = incoming
                .filter(|x| x.0.et == FirEdgeType::MuxFalse).collect();

            let true_ep  = *true_edge_vec.get(0).unwrap();
            let false_ep = *false_edge_vec.get(0).unwrap();
            let true_node  = rg.graph.node_weight(true_ep.1.0).unwrap().clone();
            let false_node = rg.graph.node_weight(false_ep.1.0).unwrap();

            assert!(
                TypeTree::eq(
                    &true_node.ttree.as_ref().unwrap(),
                    &false_node.ttree.as_ref().unwrap()),
                "Mux true and false drivers have different types {:?} {:?} vs {:?} {:?}",
                true_node,
                true_node.ttree,
                false_node,
                false_node.ttree);

            let node_mut = rg.graph.node_weight_mut(id).unwrap();
            node_mut.ttree = true_node.ttree;
        }
        FirNodeType::PrimOp2Expr(..) => {
            let incoming = rg.graph.edges_directed(id, Incoming)
                .map(|x|
                    (
                        rg.graph.edge_weight(x.id()).unwrap(),
                        rg.graph.edge_endpoints(x.id()).unwrap()
                    )
                );

            let op0_edge_vec: Vec<EdgeWeightEpTuple> = incoming.clone()
                .filter(|x| x.0.et == FirEdgeType::Operand0).collect();

            let op1_edge_vec: Vec<EdgeWeightEpTuple> = incoming.clone()
                .filter(|x| x.0.et == FirEdgeType::Operand1).collect();

            let op0_ep = *op0_edge_vec.get(0).unwrap();
            let op1_ep = *op1_edge_vec.get(0).unwrap();
            let op0_node = rg.graph.node_weight(op0_ep.1.0).unwrap().clone();
            let op1_node = rg.graph.node_weight(op1_ep.1.0).unwrap();

            assert!(
                TypeTree::eq(
                    &op0_node.ttree.as_ref().unwrap(),
                    &op1_node.ttree.as_ref().unwrap()),
                "Primop2Expr op0 and op1 drivers have different types {:?} {:?} vs {:?} {:?}",
                op0_node,
                op0_node.ttree,
                op1_node,
                op1_node.ttree);

            let node_mut = rg.graph.node_weight_mut(id).unwrap();
            node_mut.ttree = op0_node.ttree;
        }
        FirNodeType::PrimOp1Expr(op) => {
            let incoming = rg.graph.edges_directed(id, Incoming)
                .map(|x|
                    (
                        rg.graph.edge_weight(x.id()).unwrap(),
                        rg.graph.edge_endpoints(x.id()).unwrap()
                    )
                );

            let op0_edge_vec: Vec<EdgeWeightEpTuple> = incoming
                .filter(|x| x.0.et == FirEdgeType::Operand0).collect();

            let op0_ep = *op0_edge_vec.get(0).unwrap();
            let op0_node = rg.graph.node_weight(op0_ep.1.0).unwrap().clone();

            let node_mut = rg.graph.node_weight_mut(id).unwrap();

            match op {
                PrimOp1Expr::AsUInt => {
                    set_ground_type(rg, id, GroundType::UInt);
                }
                PrimOp1Expr::AsSInt => {
                    set_ground_type(rg, id, GroundType::SInt);
                }
                PrimOp1Expr::AsClock => {
                    set_ground_type(rg, id, GroundType::Clock);
                }
                PrimOp1Expr::AsAsyncReset => {
                    set_ground_type(rg, id, GroundType::AsyncReset);
                }
                _ => {
                    node_mut.ttree = op0_node.ttree;
                }
            }
        }
        FirNodeType::PrimOp1Expr1Int(..) |
            FirNodeType::PrimOp1Expr2Int(..) => {
            let incoming = rg.graph.edges_directed(id, Incoming)
                .map(|x|
                    (
                        rg.graph.edge_weight(x.id()).unwrap(),
                        rg.graph.edge_endpoints(x.id()).unwrap()
                    )
                );

            let op0_edge_vec: Vec<EdgeWeightEpTuple> = incoming
                .filter(|x| x.0.et == FirEdgeType::Operand0).collect();

            let op0_ep = *op0_edge_vec.get(0).unwrap();
            let op0_node = rg.graph.node_weight(op0_ep.1.0).unwrap().clone();

            let node_mut = rg.graph.node_weight_mut(id).unwrap();
            node_mut.ttree = op0_node.ttree;
        }
        _ => {
            panic!("Called infer_typetree_node on unexpected node type {:?}", node);
        }
    }
}
