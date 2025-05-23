use std::collections::VecDeque;
use rusty_firrtl::*;
use petgraph::Direction::{Incoming, Outgoing};
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeRef, VisitMap, Visitable};
use petgraph::Undirected;
use petgraph::prelude::Dfs;
use indexmap::IndexMap;
use crate::common::graphviz::DefaultGraphVizCore;
use crate::common::RippleIRErr;
use crate::ir::fir::*;
use crate::ir::hierarchy::Hierarchy;
use crate::ir::typetree::typetree::*;
use crate::ir::typetree::tnode::*;

pub fn infer_typetree(ir: &mut FirIR) {
    let hier = Hierarchy::new(ir);

    for module in hier.topo_order() {
        infer_typetree_graph(ir, module.name());
    }
}

fn set_ground_type(fg: &mut FirGraph, id: NodeIndex, gt: GroundType) {
    let node_mut = fg.graph.node_weight_mut(id).unwrap();
    node_mut.ttree = Some(TypeTree::build_from_ground_type(gt));
}

fn topo_start_node(nt: &FirNodeType) -> bool {
    match nt {
        FirNodeType::Invalid              |
            FirNodeType::DontCare         |
            FirNodeType::UIntLiteral(..)  |
            FirNodeType::SIntLiteral(..)  |
            FirNodeType::Wire             |
            FirNodeType::Reg              |
            FirNodeType::RegReset         |
            FirNodeType::SMem(..)         |
            FirNodeType::CMem             |
            FirNodeType::ReadMemPort(..)  |
            FirNodeType::WriteMemPort(..) |
            FirNodeType::InferMemPort(..) |
            FirNodeType::Inst(..)         |
            FirNodeType::Input            |
            FirNodeType::Output           |
            FirNodeType::Phi(..) => {
                return true;
            }
        _ => {
            return false;
        }
    }
}

fn infer_typetree_graph(fir: &mut FirIR, name: &Identifier) {
    use crate::ir::typetree::tnode::GroundType::*;

    let node_ids: Vec<NodeIndex> = fir.graphs.get(name).unwrap().node_indices().collect();

    // Take care of nodes with ground types and memory
    for &id in node_ids.iter() {
        let fg = fir.graphs.get(name).unwrap();
        let nt = &fg.graph.node_weight(id).unwrap().nt;
        match nt {
            &FirNodeType::Invalid         => { set_ground_type(fir.graphs.get_mut(name).unwrap(), id, Invalid); }
            &FirNodeType::DontCare        => { set_ground_type(fir.graphs.get_mut(name).unwrap(), id, DontCare); }
            &FirNodeType::UIntLiteral(w, ..) => { set_ground_type(fir.graphs.get_mut(name).unwrap(), id, UInt(Some(w.clone()))); }
            &FirNodeType::SIntLiteral(w, ..) => { set_ground_type(fir.graphs.get_mut(name).unwrap(), id, SInt(Some(w.clone()))); }
            FirNodeType::Inst(child_module_name)        => {
                let child_module_name = child_module_name.clone();
                let cg = fir.graphs.get(&child_module_name).unwrap();
                let mut cg_io_ttree = cg.io_typetree();
                cg_io_ttree.flip();

                let node_mut = fir.graphs.get_mut(name).unwrap().graph.node_weight_mut(id).unwrap();
                node_mut.ttree = Some(cg_io_ttree);
            }
            FirNodeType::CMem |
                FirNodeType::SMem(..) => {
                let ttree = fg.graph.node_weight(id).unwrap().ttree.clone();
                let ttree_view = ttree.as_ref().unwrap().view().unwrap();

                // Copy ttree to each port
                let childs: Vec<NodeIndex> = fg.graph.neighbors_directed(id, Outgoing).collect();
                for cid in childs {
                    let child = fir.graphs.get_mut(name).unwrap().graph.node_weight_mut(cid).unwrap();
                    match child.nt {
                        FirNodeType::InferMemPort(..) |
                            FirNodeType::WriteMemPort(..) |
                            FirNodeType::ReadMemPort(..) => {
                            child.ttree = Some(ttree_view.subtree_array_element().clone_ttree());
                        }
                        _ => {
                            panic!("{:?} Unexpected child node {:?} for memory", name, child);
                        }
                    }
                }
            }
            _ => {
            }
        }
    }
    for &id in node_ids.iter() {
        let fg = fir.graphs.get(name).unwrap();
        let nt = &fg.graph.node_weight(id).unwrap().nt;
        match nt {
            FirNodeType::Phi(..) => {
                let childs: Vec<NodeIndex> = fg.graph.neighbors_directed(id, Outgoing).collect();
                assert!(childs.len() == 1, "Phi node should have a single child {:?}", name);

                let cid = childs[0];
                let mut child_ttree = fg.graph.node_weight(cid).unwrap().ttree.clone();
                let child = fg.graph.node_weight(cid).unwrap();
                let child_is_io_or_inst = match child.nt {
                    FirNodeType::Input |
                        FirNodeType::Output |
                        FirNodeType::Inst(..) => {
                            true
                    }
                    _ => false,
                };

                let node_mut = fir.graphs.get_mut(name).unwrap().graph.node_weight_mut(id).unwrap();

                if child_is_io_or_inst {
                    child_ttree.as_mut().unwrap().flip();
                }
                node_mut.ttree = child_ttree;
                assert!(node_mut.ttree.is_some());
            }
            _ => {
            }
        }
    }

    let fg = fir.graphs.get_mut(name).unwrap();

    // compute indeg for the entire graph
    let mut indeg: IndexMap<NodeIndex, u32> = IndexMap::new();
    for id in fg.graph.node_indices() {
        indeg.insert(id, 0);
    }
    for eid in fg.graph.edge_indices() {
        let ep = fg.graph.edge_endpoints(eid).unwrap();
        let dst = ep.1;
        *indeg.get_mut(&dst).unwrap() += 1;
    }

    // Main type inference loop
    // - Topo sort nodes in each CC, and perform type inference
    let undir_graph = fg.graph.clone().into_edge_type::<Undirected>();
    let mut visited = 0;
    let mut vis_map = fg.graph.visit_map();
    let mut topo_vismap = fg.graph.visit_map();
    for id in fg.graph.node_indices() {
        if vis_map.is_visited(&id) {
            continue;
        }

        let mut q: VecDeque<NodeIndex> = VecDeque::new();
        let mut dfs = Dfs::new(&undir_graph, id);
        while let Some(nx) = dfs.next(&undir_graph) {
            vis_map.visit(nx);

            let node = fg.graph.node_weight(nx).unwrap();
            if topo_start_node(&node.nt) {
                q.push_back(nx);
            }
        }

        // Start topological sort
        let mut topo_sort_order: Vec<NodeIndex> = vec![];
        let mut topo_vis_map = fg.graph.visit_map();
        while !q.is_empty() {
            let nidx = q.pop_front().unwrap();
            if topo_vis_map.is_visited(&nidx) {
                continue;
            }

            topo_vis_map.visit(nidx);
            topo_sort_order.push(nidx);

            let childs = fg.graph.neighbors_directed(nidx, Outgoing);
            for cidx in childs {
                let child = fg.graph.node_weight(cidx).unwrap();
                if !topo_vis_map.is_visited(&cidx) && !topo_start_node(&child.nt) {
                    *indeg.get_mut(&cidx).unwrap() -= 1;
                    if *indeg.get(&cidx).unwrap() == 0 {
                        q.push_back(cidx);
                    }
                }
            }
        }

        // Infer type in topo sorted order
        for nidx in topo_sort_order.iter() {
            topo_vismap.visit(*nidx);
            let node = fg.graph.node_weight(*nidx).unwrap();
            if node.ttree.is_some() {
                continue;
            }
            infer_typetree_node(fg, *nidx, name);
        }

        visited += topo_sort_order.len();
    }

    if visited != vis_map.len() {
        for id in fg.graph.node_indices() {
            if !topo_vismap.is_visited(&id) {
                let unvisited = fg.graph.node_weight(id).unwrap();
                println!("Unvisited {:?}", unvisited);
            }
        }
        assert!(
            false,
            "{:?}: visited {} nodes out of {} nodes while topo sorting",
            name,
            visited,
            vis_map.len());
    }
}

fn infer_typetree_node(fg: &mut FirGraph, id: NodeIndex, name: &Identifier) {
    let node = fg.graph.node_weight(id).unwrap();
    if node.ttree.is_some() {
        panic!("{:?}: infer typetree node overriding {:?}", name, node);
    }

    type EdgeEndpoint = (NodeIndex, NodeIndex);
    type EdgeWeightEpTuple<'a> = (&'a FirEdge, EdgeEndpoint);

    match node.nt {
        FirNodeType::Mux => {
            let incoming = fg.graph.edges_directed(id, Incoming)
                .map(|x|
                    (
                        fg.graph.edge_weight(x.id()).unwrap(),
                        fg.graph.edge_endpoints(x.id()).unwrap()
                    )
                );
            let true_edge_vec: Vec<EdgeWeightEpTuple> = incoming.clone()
                .filter(|x| x.0.et == FirEdgeType::MuxTrue).collect();

            let false_edge_vec: Vec<EdgeWeightEpTuple> = incoming
                .filter(|x| x.0.et == FirEdgeType::MuxFalse).collect();

            let true_ep  = *true_edge_vec.get(0).unwrap();
            let false_ep = *false_edge_vec.get(0).unwrap();
            let true_node  = fg.graph.node_weight(true_ep.1.0).unwrap().clone();
            let false_node = fg.graph.node_weight(false_ep.1.0).unwrap();

            let true_ttree_view = true_node.ttree.as_ref().unwrap().view().unwrap();
            let true_ttree_subtree = true_ttree_view.subtree_from_expr(&true_ep.0.src).unwrap();

            let false_ttree_view = false_node.ttree.as_ref().unwrap().view().unwrap();
            let false_ttree_subtree = false_ttree_view.subtree_from_expr(&false_ep.0.src).unwrap();

            if !true_ttree_subtree.eq(&false_ttree_subtree) {
                true_ttree_subtree.print_tree();
                false_ttree_subtree.print_tree();
                panic!("{:?}: Mux true and false drivers have different types {:?} {:?} {:?} {:?}",
                    name,
                    true_node,
                    true_ep.0,
                    false_node,
                    false_ep.0);
            }

            let node_mut = fg.graph.node_weight_mut(id).unwrap();
            node_mut.ttree = Some(true_ttree_subtree.clone_ttree());
        }
        FirNodeType::PrimOp2Expr(op) => {
            let incoming = fg.graph.edges_directed(id, Incoming)
                .map(|x|
                    (
                        fg.graph.edge_weight(x.id()).unwrap(),
                        fg.graph.edge_endpoints(x.id()).unwrap()
                    )
                );

            let op0_edge_vec: Vec<EdgeWeightEpTuple> = incoming.clone()
                .filter(|x| x.0.et == FirEdgeType::Operand0).collect();

            let op1_edge_vec: Vec<EdgeWeightEpTuple> = incoming.clone()
                .filter(|x| x.0.et == FirEdgeType::Operand1).collect();

            let op0_ep = *op0_edge_vec.get(0).unwrap();
            let op1_ep = *op1_edge_vec.get(0).unwrap();
            let op0_node = fg.graph.node_weight(op0_ep.1.0).unwrap().clone();
            let op1_node = fg.graph.node_weight(op1_ep.1.0).unwrap();

            let op0_type_tree_view = op0_node.ttree.as_ref().unwrap().view().unwrap();
            let op0_type_tree = op0_type_tree_view.subtree_from_expr(&op0_ep.0.src).unwrap();

            let op1_type_tree_view = op1_node.ttree.as_ref().unwrap().view().unwrap();
            let op1_type_tree = op1_type_tree_view.subtree_from_expr(&op1_ep.0.src).unwrap();

            if !op0_type_tree.eq(&op1_type_tree) && op != PrimOp2Expr::Dshr {
                op0_type_tree_view.subtree_from_expr(&op0_ep.0.src).unwrap().print_tree();
                op0_type_tree.print_tree();
                op1_type_tree.print_tree();
                panic!("{:?}: PrimOp2Expr op0 and op1 drivers have different types {:?} {:?} {:?} {:?}",
                    name,
                    op0_node,
                    op0_ep.0,
                    op1_node,
                    op1_ep.0);
            }

            match op {
                PrimOp2Expr::Eq  |
                PrimOp2Expr::Neq |
                PrimOp2Expr::Lt  |
                PrimOp2Expr::Leq |
                PrimOp2Expr::Gt  |
                PrimOp2Expr::Geq |
                PrimOp2Expr::And |
                PrimOp2Expr::Or  |
                PrimOp2Expr::Xor |
                PrimOp2Expr::Cat => {
                    assert!(op0_type_tree.is_ground_type());
                    set_ground_type(fg, id, GroundType::UInt(None));
                }
                _ => {
                    let node_mut = fg.graph.node_weight_mut(id).unwrap();
                    node_mut.ttree = Some(op0_type_tree.clone_ttree());
                }
            }

        }
        FirNodeType::PrimOp1Expr(op) => {
            let incoming = fg.graph.edges_directed(id, Incoming)
                .map(|x|
                    (
                        fg.graph.edge_weight(x.id()).unwrap(),
                        fg.graph.edge_endpoints(x.id()).unwrap()
                    )
                );

            let op0_edge_vec: Vec<EdgeWeightEpTuple> = incoming.clone()
                .filter(|x| x.0.et == FirEdgeType::Operand0).collect();

            if op0_edge_vec.len() != 1 {
                eprintln!("{:?} doesn't have enough incoming edges {:?} {:?}",
                    name,
                    node,
                    incoming.collect::<Vec<EdgeWeightEpTuple>>());
            }

            let op0_ep = *op0_edge_vec.get(0).unwrap();
            let op0_node = fg.graph.node_weight(op0_ep.1.0).unwrap().clone();
            let op0_type_tree_view = op0_node.ttree.as_ref().unwrap().view().unwrap();
            let op0_type_tree = op0_type_tree_view.subtree_from_expr(&op0_ep.0.src).unwrap();

            match op {
                PrimOp1Expr::AsUInt => {
                    assert!(op0_type_tree.is_ground_type());
                    set_ground_type(fg, id, GroundType::UInt(None));
                }
                PrimOp1Expr::AsSInt => {
                    assert!(op0_type_tree.is_ground_type());
                    set_ground_type(fg, id, GroundType::SInt(None));
                }
                PrimOp1Expr::AsClock => {
                    assert!(op0_type_tree.is_ground_type());
                    set_ground_type(fg, id, GroundType::Clock);
                }
                PrimOp1Expr::AsAsyncReset => {
                    assert!(op0_type_tree.is_ground_type());
                    set_ground_type(fg, id, GroundType::AsyncReset);
                }
                PrimOp1Expr::Cvt => {
                    assert!(op0_type_tree.is_ground_type());
                    set_ground_type(fg, id, GroundType::SInt(None));
                }
                PrimOp1Expr::Not  |
                PrimOp1Expr::Andr |
                PrimOp1Expr::Orr  |
                PrimOp1Expr::Xorr => {
                    assert!(op0_type_tree.is_ground_type());
                    set_ground_type(fg, id, GroundType::UInt(None));
                }
                _ => {
                    let node_mut = fg.graph.node_weight_mut(id).unwrap();
                    node_mut.ttree = Some(op0_type_tree.clone_ttree());
                }
            }
        }
        FirNodeType::PrimOp1Expr1Int(op, ..) => {
            let incoming = fg.graph.edges_directed(id, Incoming)
                .map(|x|
                    (
                        fg.graph.edge_weight(x.id()).unwrap(),
                        fg.graph.edge_endpoints(x.id()).unwrap()
                    )
                );

            let op0_edge_vec: Vec<EdgeWeightEpTuple> = incoming
                .filter(|x| x.0.et == FirEdgeType::Operand0).collect();

            let op0_ep = *op0_edge_vec.get(0).unwrap();
            let op0_node = fg.graph.node_weight(op0_ep.1.0).unwrap().clone();
            let op0_type_tree_view = op0_node.ttree.as_ref().unwrap().view().unwrap();
            let op0_type_tree = op0_type_tree_view.subtree_from_expr(&op0_ep.0.src).unwrap();

            assert!(op0_type_tree.is_ground_type());
            match op {
                PrimOp1Expr1Int::Head |
                    PrimOp1Expr1Int::Tail => {
                        assert!(op0_type_tree.is_ground_type());
                        set_ground_type(fg, id, GroundType::UInt(None));
                }
                _ => {
                    let node_mut = fg.graph.node_weight_mut(id).unwrap();
                    node_mut.ttree = Some(op0_type_tree.clone_ttree());
                }
            }

        }
        FirNodeType::PrimOp1Expr2Int(..) => {
            let incoming = fg.graph.edges_directed(id, Incoming)
                .map(|x|
                    (
                        fg.graph.edge_weight(x.id()).unwrap(),
                        fg.graph.edge_endpoints(x.id()).unwrap()
                    )
                );

            let op0_edge_vec: Vec<EdgeWeightEpTuple> = incoming
                .filter(|x| x.0.et == FirEdgeType::Operand0).collect();

            let op0_ep = *op0_edge_vec.get(0).unwrap();
            let op0_node = fg.graph.node_weight(op0_ep.1.0).unwrap().clone();
            let op0_type_tree_view = op0_node.ttree.as_ref().unwrap().view().unwrap();
            let op0_type_tree = op0_type_tree_view.subtree_from_expr(&op0_ep.0.src).unwrap();

            assert!(op0_type_tree.is_ground_type());
            set_ground_type(fg, id, GroundType::UInt(None));
        }
        FirNodeType::Printf(..) |
            FirNodeType::Assert(..) => {
        }
        _ => {
            panic!("{:?}: Called infer_typetree_node on unexpected node type {:?}", name, node);
        }
    }
}

pub fn check_typetree_inference(ir: &FirIR) -> Result<(), RippleIRErr> {
    for (_name, fg) in ir.graphs.iter() {
        check_typetree_inference_graph(fg)?;
    }
    return Ok(());
}

fn check_typetree_inference_graph(fg: &FirGraph) -> Result<(), RippleIRErr> {
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
        match node.nt {
            FirNodeType::Printf(..) |
                FirNodeType::Assert(..) => {
                    continue;
            }
            _ => {
                if node.ttree.is_none() {
                    return Err(
                        RippleIRErr::TypeTreeInferenceError(
                            format!("{:?} does not have a typetree", node.clone())));
                }
            }
        }
    }
    return Ok(());
}

#[cfg(test)]
mod infer_typetree_test {
    use crate::common::RippleIRErr;
    use crate::passes::runner::run_passes_from_filepath;

    fn check_run_to_completion(input: &str) -> Result<(), RippleIRErr> {
        let _fir = run_passes_from_filepath(input)?;
        Ok(())
    }

    #[test]
    fn gcd() {
        check_run_to_completion("./test-inputs/GCD.fir")
            .expect("gcd to run to completion");
    }

    #[test]
    fn decoupledmux() {
        check_run_to_completion("./test-inputs/DecoupledMux.fir")
            .expect("decoupledmux to run to completion");
    }

    #[test]
    fn hierarchy() {
        check_run_to_completion("./test-inputs/Hierarchy.fir")
            .expect("hierarchy to run to completion");
    }

    #[test]
    fn rocket() {
        check_run_to_completion("./test-inputs/chipyard.harness.TestHarness.RocketConfig.fir")
            .expect("rocket to run to completion");
    }

    #[test]
    fn boom() {
        check_run_to_completion("./test-inputs/chipyard.harness.TestHarness.LargeBoomV3Config.fir")
            .expect("boom to run to completion");
    }
}
