use crate::common::graphviz::DefaultGraphVizCore;
use crate::ir::fir::{FirEdgeType, FirGraph, FirIR, FirNodeType};
use crate::ir::whentree::{Condition, Conditions, PhiPriority, PrioritizedCond, WhenTree, WhenTreeNode};
use chirrtl_parser::ast::*;
use std::collections::VecDeque;
use indexmap::IndexMap;
use petgraph::{visit::EdgeRef, Direction::Incoming, Direction::Outgoing};
use petgraph::graph::NodeIndex;
use petgraph::Undirected;
use petgraph::visit::VisitMap;
use petgraph::visit::Visitable;
use petgraph::prelude::Dfs;

pub fn to_ast(fir: &FirIR) -> Circuit {
    let mut cms = CircuitModules::new();
    for (name, fg) in fir.graphs.iter() {
        let cm = to_circuitmodule(name, fg);
        cms.push(Box::new(cm));

    }
    Circuit::new(fir.version.clone(), fir.name.clone(), fir.annos.clone(), cms)
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
    collect_op_stmts(fg, &mut stmts);
    collect_conn_stmts(fg, &mut stmts);
    Module::new(name.clone(), ports, stmts, Info::default())
}

/// Statements that defines a structural element
fn collect_def_stmts(fg: &FirGraph, stmts: &mut Stmts) {
    for id in fg.node_indices() {
        let node = fg.node_weight(id).unwrap();
        let tpe = node.ttree.as_ref().unwrap().to_type();
        match &node.nt {
            FirNodeType::Wire => {
                let name = node.name.as_ref().unwrap().clone();
                stmts.push(Box::new(Stmt::Wire(name, tpe, Info::default())));
            }
            FirNodeType::Reg => {
                let clk_eid = fg.parent_with_type(id, FirEdgeType::Clock).unwrap();
                let clk = fg.graph.edge_weight(clk_eid).unwrap().src.clone();

                let name = node.name.as_ref().unwrap().clone();
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

                let name = node.name.as_ref().unwrap().clone();
                let reg_init = Stmt::RegReset(name, tpe, clk, rst, init, Info::default());
                stmts.push(Box::new(reg_init));
            }
            FirNodeType::SMem(ruw_opt) => {
                let name = node.name.as_ref().unwrap().clone();
                let smem = ChirrtlMemory::SMem(name, tpe, ruw_opt.clone(), Info::default());
                let mem = Stmt::ChirrtlMemory(smem);
                stmts.push(Box::new(mem));
            }
            FirNodeType::CMem => {
                let name = node.name.as_ref().unwrap().clone();
                let cmem = ChirrtlMemory::CMem(name, tpe, Info::default());
                let mem = Stmt::ChirrtlMemory(cmem);
                stmts.push(Box::new(mem));
            }
            FirNodeType::Inst(module) => {
                let name = node.name.as_ref().unwrap().clone();
                let inst = Stmt::Inst(name, module.clone(), Info::default());
                stmts.push(Box::new(inst));
            }
            _ => {
                continue;
            }
        }
    }
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
            FirNodeType::Phi => {
                return true;
            }
        _ => {
            return false;
        }
    }
}

/// Statements that performs some operation
fn collect_op_stmts(fg: &FirGraph, stmts: &mut Stmts) {
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
    // - Topo sort nodes in each CC
    let undir_graph = fg.graph.clone().into_edge_type::<Undirected>();
    let mut vis_map = fg.graph.visit_map();
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

        // Add Stmt if it is a operation
        for nidx in topo_sort_order.iter() {
            collect_op_stmt(fg, stmts, *nidx);
        }
    }
}

fn collect_op_stmt(fg: &FirGraph, stmts: &mut Stmts, id: NodeIndex) {
    let node = fg.graph.node_weight(id).unwrap();
    match &node.nt {
        FirNodeType::Mux => {
            let true_eid  = fg.parent_with_type(id, FirEdgeType::MuxTrue).unwrap();
            let false_eid = fg.parent_with_type(id, FirEdgeType::MuxFalse).unwrap();
            let cond_eid  = fg.parent_with_type(id, FirEdgeType::MuxCond).unwrap();

            let t = fg.graph.edge_weight(true_eid).unwrap().src.clone();
            let f = fg.graph.edge_weight(false_eid).unwrap().src.clone();
            let c = fg.graph.edge_weight(cond_eid).unwrap().src.clone();
            let mux = Expr::Mux(Box::new(c), Box::new(t), Box::new(f));

            let name = node.name.as_ref().unwrap().clone();
            let stmt = Stmt::Node(name, mux, Info::default());
            stmts.push(Box::new(stmt));
        }
        FirNodeType::PrimOp2Expr(op) => {
            let op0_eid = fg.parent_with_type(id, FirEdgeType::Operand0).unwrap();
            let op1_eid = fg.parent_with_type(id, FirEdgeType::Operand1).unwrap();

            let op0 = fg.graph.edge_weight(op0_eid).unwrap().src.clone();
            let op1 = fg.graph.edge_weight(op1_eid).unwrap().src.clone();
            let primop = Expr::PrimOp2Expr(op.clone(), Box::new(op0), Box::new(op1));

            let name = node.name.as_ref().unwrap().clone();
            let stmt = Stmt::Node(name, primop, Info::default());
            stmts.push(Box::new(stmt));
        }
        FirNodeType::PrimOp1Expr(op) => {
            if let Some(name) = node.name.as_ref() {
                let op0_eid = fg.parent_with_type(id, FirEdgeType::Operand0).unwrap();
                let op0 = fg.graph.edge_weight(op0_eid).unwrap().src.clone();
                let primop = Expr::PrimOp1Expr(op.clone(), Box::new(op0));

                let stmt = Stmt::Node(name.clone(), primop, Info::default());
                stmts.push(Box::new(stmt));
            }
        }
        FirNodeType::PrimOp1Expr1Int(op, x) => {
            let op0_eid = fg.parent_with_type(id, FirEdgeType::Operand0).unwrap();
            let op0 = fg.graph.edge_weight(op0_eid).unwrap().src.clone();
            let primop = Expr::PrimOp1Expr1Int(op.clone(), Box::new(op0), x.clone());

            let name = node.name.as_ref().unwrap().clone();
            let stmt = Stmt::Node(name, primop, Info::default());
            stmts.push(Box::new(stmt));
        }
        FirNodeType::PrimOp1Expr2Int(op, x, y) => {
            let op0_eid = fg.parent_with_type(id, FirEdgeType::Operand0).unwrap();
            let op0 = fg.graph.edge_weight(op0_eid).unwrap().src.clone();
            let primop = Expr::PrimOp1Expr2Int(op.clone(), Box::new(op0), x.clone(), y.clone());

            let name = node.name.as_ref().unwrap().clone();
            let stmt = Stmt::Node(name, primop, Info::default());
            stmts.push(Box::new(stmt));
        }
        _ => {  }
    }
}

fn reconstruct_whentree(fg: &FirGraph) -> WhenTree {
    let mut cond_priority_pair = vec![];
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
        if node.nt == FirNodeType::Phi {
            let eids = fg.graph.edges_directed(id, Incoming);
            for eid in eids {
                let edge = fg.graph.edge_weight(eid.id()).unwrap();
                if let FirEdgeType::PhiInput(prior, cond) = &edge.et {
                    let pc = PrioritizedCond::new(prior.clone(), cond.clone());
                    cond_priority_pair.push(pc);
                }
            }
        }
    }
    WhenTree::from_conditions(cond_priority_pair)
}

fn insert_memport_stmts(fg: &FirGraph, whentree: &mut WhenTree) {
    // Add memory port definition statements
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
        match &node.nt {
            FirNodeType::ReadMemPort(conds)      |
                FirNodeType::WriteMemPort(conds) |
                FirNodeType::InferMemPort(conds) => {
                let clk_eid  = fg.parent_with_type(id, FirEdgeType::Clock).unwrap();
                let addr_eid = fg.parent_with_type(id, FirEdgeType::MemPortAddr).unwrap();
                let mem_eid = fg.parent_with_type(id, FirEdgeType::MemPortEdge).unwrap();

                let mem = &fg.graph.edge_weight(mem_eid).unwrap().src;
                let clk = &fg.graph.edge_weight(clk_eid).unwrap().src;
                let addr = fg.graph.edge_weight(addr_eid).unwrap().src.clone();

                let port_stmt = if let (
                    Expr::Reference(Reference::Ref(mem_name)),
                    Expr::Reference(clk_ref)
                ) = (mem, clk) {
                    let name = node.name.as_ref().unwrap().clone();

                    let port = match &node.nt {
                        FirNodeType::ReadMemPort(..) => {
                            ChirrtlMemoryPort::Read(name, mem_name.clone(),
                                addr, clk_ref.clone(), Info::default())
                        }
                        FirNodeType::WriteMemPort(..) => {
                            ChirrtlMemoryPort::Write(name, mem_name.clone(),
                                addr, clk_ref.clone(), Info::default())
                        }
                        FirNodeType::InferMemPort(..) => {
                            ChirrtlMemoryPort::Infer(name, mem_name.clone(),
                                addr, clk_ref.clone(), Info::default())
                        }
                        _ => { unreachable!(); }
                    };
                    Stmt::ChirrtlMemoryPort(port)
                } else {
                    panic!("Unrecognized MPORT with memory {:?} and clk {:?}", mem, clk);
                };

                // Priority is set to `None`
                // This can change the port position when the port has no
                // enable signal (from the very end of the module to the very top.
                // However, this shouldn't affect the behavior anyways
                let when_leaf = whentree.get_node_mut(conds, None).unwrap();
                when_leaf.stmts.push(Box::new(port_stmt));
            }
            _ => {
                continue;
            }
        }
    }
}

fn insert_conn_stmts(fg: &FirGraph, whentree: &mut WhenTree) {
    let mut ordered_stmts: IndexMap<&Conditions, Vec<(PhiPriority, Stmt)>> = IndexMap::new();
    for eid in fg.graph.edge_indices() {
        let edge = fg.graph.edge_weight(eid).unwrap();
        if edge.dst.is_none() {
            continue;
        }

        let lhs = Expr::Reference(edge.dst.as_ref().unwrap().clone());
        let rhs = edge.src.clone();
        let stmt = Stmt::Connect(lhs, rhs, Info::default());

        match &edge.et {
            FirEdgeType::PhiInput(prior, conds) => {
                if !ordered_stmts.contains_key(conds) {
                    ordered_stmts.insert(conds, vec![]);
                }
                ordered_stmts.get_mut(conds).unwrap().push((prior.clone(), stmt));
            }
            _ => {
                let leaf = whentree.get_node_mut(&Conditions::root(), None).unwrap();
                leaf.stmts.push(Box::new(stmt));
            }
        }
    }
    for stmts in ordered_stmts.values_mut() {
        stmts.sort_by(|a, b| b.0.cmp(&a.0));
    }

    for (conds, prior_stmts) in ordered_stmts {
        for (prior, stmt) in prior_stmts {
            let leaf = whentree.get_node_mut(conds, Some(&prior)).unwrap();
            leaf.stmts.push(Box::new(stmt));
        }
    }
}

fn collect_conn_stmts(fg: &FirGraph, stmts: &mut Stmts) {
    let mut whentree = reconstruct_whentree(fg);
    insert_memport_stmts(fg, &mut whentree);
    insert_conn_stmts(fg, &mut whentree);
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::passes::fir::from_ast::from_circuit;
    use crate::passes::fir::remove_unnecessary_phi::remove_unnecessary_phi;
    use crate::passes::fir::check_phi_nodes::check_phi_node_connections;
    use crate::common::RippleIRErr;
    use chirrtl_parser::parse_circuit;
    use test_case::test_case;
    use indexmap::IndexMap;

    #[test_case("GCD" ; "GCD")]
    #[test_case("NestedWhen" ; "NestedWhen")]
    #[test_case("LCS1" ; "LCS1")]
    #[test_case("LCS2" ; "LCS2")]
    #[test_case("LCS3" ; "LCS3")]
    #[test_case("LCS4" ; "LCS4")]
    #[test_case("LCS5" ; "LCS5")]
    #[test_case("LCS6" ; "LCS6")]
    #[test_case("LCS7" ; "LCS7")]
    #[test_case("LCS8" ; "LCS8")]
    fn run(name: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(format!("./test-inputs/{}.fir", name))?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let mut ast_whentrees: IndexMap<&Identifier, WhenTree> = IndexMap::new();
        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    let mut whentree = WhenTree::new();
                    whentree.from_stmts(&m.stmts);
                    ast_whentrees.insert(&m.name, whentree);
                }
                CircuitModule::ExtModule(_) => {
                    continue;
                }
            }
        }

        let mut ir = from_circuit(&circuit);
        remove_unnecessary_phi(&mut ir);
        check_phi_node_connections(&ir)?;

        for (name, fg) in ir.graphs.iter() {
            if ast_whentrees.contains_key(name) {
                let ast_whentree = ast_whentrees.get(name).unwrap();
                let ast_leaves = ast_whentree.leaf_to_conditions();

                let fir_whentree = reconstruct_whentree(fg);
                let fir_leaves = fir_whentree.leaf_to_conditions();

                fir_whentree.print_tree();

                for (fnode, fconds) in fir_leaves {
                    assert!(ast_leaves.contains_key(fnode));
                    let aconds = ast_leaves.get(fnode).unwrap();
                    assert_eq!(aconds, &fconds);
                }
            }
        }
        Ok(())
    }

    // TODO: Stuff to check
    // (More) Last connect semantics
    // Memory
    // Invalid (i.e. DontCare)
}
