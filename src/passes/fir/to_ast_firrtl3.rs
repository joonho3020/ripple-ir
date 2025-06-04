use rusty_firrtl::*;
use indexmap::IndexMap;
use indexmap::IndexSet;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::{visit::EdgeRef, Direction::Incoming, Direction::Outgoing};
use petgraph::visit::VisitMap;
use petgraph::visit::Visitable;
use petgraph::prelude::Dfs;
use petgraph::Undirected;
use std::collections::VecDeque;
use crate::common::graphviz::DefaultGraphVizCore;
use crate::ir::fir::FirIR;
use crate::passes::fir::to_ast::get_ports;
use crate::passes::fir::to_ast::find_array_addr_chain_in_ref;
use crate::passes::fir::to_ast::reverse_adjacency_list;
use crate::passes::fir::to_ast::reconstruct_whentree;
use crate::passes::fir::to_ast::insert_op_stmts;
use crate::passes::fir::to_ast::insert_invalidate_stmts;
use crate::passes::fir::to_ast::insert_def_topo_start_stmts;
use crate::passes::fir::to_ast::insert_printf_assertion_stmts;
use crate::passes::fir::to_ast::insert_conn_stmts;
use crate::passes::fir::to_ast::find_highest_path;
use crate::passes::fir::to_ast::EmissionInfo;
use crate::ir::fir::{FirEdgeType, FirGraph, FirNodeType};
use crate::ir::whentree::{CondPath, StmtWithPrior, WhenTree};

pub fn to_ast_firrtl3(fir: &FirIR) -> Circuit {
    let mut cms = CircuitModules::new();
    for hier in fir.hier.topo_order() {
        let fg = fir.graphs.get(hier.name()).unwrap();
        let cm = to_circuitmodule_firrtl3(hier.name(), fg);
        cms.push(Box::new(cm));
    }
    Circuit::new(fir.version.clone(), fir.name.clone(), fir.annos.clone(), cms)
}

fn to_circuitmodule_firrtl3(name: &Identifier, fg: &FirGraph) -> CircuitModule {
    if fg.blackbox {
        CircuitModule::ExtModule(to_extmodule(name, fg))
    } else {
        CircuitModule::Module(to_module(name, fg))
    }
}

pub fn to_extmodule(name: &Identifier, fg: &FirGraph) -> ExtModule {
    let ext_info = &fg.ext_info.as_ref().unwrap();
    let defname = &ext_info.defname;
    let params = &ext_info.params;
    let ports = get_ports(fg);
    ExtModule::new(name.clone(), ports, defname.clone(), params.clone(), Info::default())
}

fn to_module(name: &Identifier, fg: &FirGraph) -> Module {
    let ports = get_ports(fg);
    let mut stmts: Stmts = Stmts::new();
    collect_stmts(fg, &mut stmts);
    Module::new(name.clone(), ports, stmts, Info::default())
}

fn topo_start_node(fg: &FirGraph, id: NodeIndex) -> bool {
    let node = fg.graph.node_weight(id).unwrap();
    match node.nt {
        FirNodeType::Invalid  |
        FirNodeType::Input    |
        FirNodeType::Output   |
        FirNodeType::UIntLiteral(..)  |
        FirNodeType::SIntLiteral(..)  |
        FirNodeType::Wire     |
        FirNodeType::Inst(..) |
        FirNodeType::Memory(..) => {
            true
        }
        _ => {
            false
        }
    }
}

fn collect_stmts(fg: &FirGraph, stmts: &mut Stmts) {
    let mut whentree = reconstruct_whentree(fg);
    let mut array_addr_childs: IndexMap<NodeIndex, IndexSet<NodeIndex>> = IndexMap::new();
    let mut invalidate_edges: IndexMap<NodeIndex, IndexSet<NodeIndex>> = IndexMap::new();

    // Compute indeg for the entire graph
    let mut indeg: IndexMap<NodeIndex, u32> = IndexMap::new();
    for id in fg.graph.node_indices() {
        indeg.insert(id, 0);
    }

    for eid in fg.graph.edge_indices() {
        let ep = fg.graph.edge_endpoints(eid).unwrap();
        let dst = ep.1;
        let dst_node = fg.graph.node_weight(dst).unwrap();
        let edge = fg.graph.edge_weight(eid).unwrap();

        if dst_node.nt == FirNodeType::Reg {
            if edge.et == FirEdgeType::Clock {
                *indeg.get_mut(&dst).unwrap() += 1;
            }
        } else if dst_node.nt == FirNodeType::RegReset {
            if edge.et == FirEdgeType::Clock || edge.et == FirEdgeType::Reset || edge.et == FirEdgeType::InitValue {
                *indeg.get_mut(&dst).unwrap() += 1;
            }
        } else {
            *indeg.get_mut(&dst).unwrap() += 1;
        }
    }

    // Add implicit dependency edges for nested references
    let mut visited_refs: IndexSet<(NodeIndex, &Reference)> = IndexSet::new();
    for id in fg.graph.node_indices() {
        let parents = fg.graph.edges_directed(id, Incoming);
        for peid in parents {
            let edge = fg.graph.edge_weight(peid.id()).unwrap();
            let ep = fg.graph.edge_endpoints(peid.id()).unwrap();
            let (src, dst) = ep;
            if let Expr::Reference(ref_expr) = &edge.src {
                if visited_refs.contains(&(dst, ref_expr)) {
                    continue;
                }
                visited_refs.insert((dst, ref_expr));
                find_array_addr_chain_in_ref(fg, src, dst, ref_expr, &whentree, &mut array_addr_childs, &mut indeg);
            }
        }
    }

    let array_addr_parents = reverse_adjacency_list(&array_addr_childs);

    // Add implicit dependency edges for invalidate stmts
    for eid in fg.graph.edge_indices() {
        let edge = fg.graph.edge_weight(eid).unwrap();
        if edge.dst.is_none() {
            continue;
        }
        let ep = fg.graph.edge_endpoints(eid).unwrap();
        let inv_id = ep.0;
        let ref_id = ep.1;

        if edge.et == FirEdgeType::DontCare {
            let ref_node = fg.graph.node_weight(ref_id).unwrap();
            let id = match ref_node.nt {
                FirNodeType::Phi(..) => {
                    let childs = fg.graph.neighbors_directed(ref_id, Outgoing);
                    childs.last().unwrap()
                }
                _ => {
                    ref_id
                }
            };
            if !invalidate_edges.contains_key(&id) {
                invalidate_edges.insert(id, IndexSet::new());
            }
            invalidate_edges.get_mut(&id).unwrap().insert(inv_id);

            assert!(indeg.get(&inv_id).unwrap() == &0);
            *indeg.get_mut(&inv_id).unwrap() += 1;
            *indeg.get_mut(&ref_id).unwrap() -= 1;
        };
    }

    let mut node_inedge_map: IndexMap<NodeIndex, IndexSet<EdgeIndex>> = IndexMap::new();

    // Topo sort nodes in each CC.
    // Must insert stmts in this order to prevent accessing into undeclared
    // variables
    let undir_graph = fg.graph.clone().into_edge_type::<Undirected>();
    let mut vis_map = fg.graph.visit_map();
    let mut visited = 0;
    let mut emission_info: EmissionInfo = EmissionInfo::default();
    for id in fg.graph.node_indices() {
        if vis_map.is_visited(&id) {
            continue;
        }

        let mut q: VecDeque<NodeIndex> = VecDeque::new();
        let mut dfs = Dfs::new(&undir_graph, id);
        let mut cc_vismap = fg.graph.visit_map();
        while let Some(nx) = dfs.next(&undir_graph) {
            vis_map.visit(nx);
            cc_vismap.visit(nx);

            if topo_start_node(&fg, nx) {
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

            if array_addr_childs.contains_key(&nidx) {
                let addr_childs = array_addr_childs.get(&nidx).unwrap();
                for &cidx in addr_childs {
                    assert!(*indeg.get(&cidx).unwrap() > 0);
                    *indeg.get_mut(&cidx).unwrap() -= 1;
                    if *indeg.get(&cidx).unwrap() == 0 {
                        q.push_back(cidx);
                    }
                }
            }

            if invalidate_edges.contains_key(&nidx) {
                let inv_childs = invalidate_edges.get(&nidx).unwrap();
                for &cidx in inv_childs {
                    assert!(*indeg.get(&cidx).unwrap() > 0);
                    *indeg.get_mut(&cidx).unwrap() -= 1;
                    if *indeg.get(&cidx).unwrap() == 0 {
                        q.push_back(cidx);
                    }
                }
            }

            let cedges = fg.graph.edges_directed(nidx, Outgoing);
            for cedge in cedges {
                let ep = fg.graph.edge_endpoints(cedge.id()).unwrap();
                if !topo_vis_map.is_visited(&ep.1) && !topo_start_node(&fg, ep.1) {
                    if !node_inedge_map.contains_key(&ep.1) {
                        node_inedge_map.insert(ep.1, IndexSet::new());
                    }
                    node_inedge_map.get_mut(&ep.1).unwrap().insert(cedge.id());
                    if *indeg.get(&ep.1).unwrap() > 0 {
                        *indeg.get_mut(&ep.1).unwrap() -= 1;
                    }
                    if *indeg.get(&ep.1).unwrap() == 0 {
                        q.push_back(ep.1);
                    }
                }
            }
        }

        if cc_vismap != topo_vis_map {
            for id in fg.graph.node_indices() {
                if cc_vismap.is_visited(&id) && !topo_vis_map.is_visited(&id) {
                    let node = fg.graph.node_weight(id).unwrap();
                    let visited_edges = node_inedge_map.get(&id).unwrap();
                    println!("- Visited during DFS Node {:?}, indeg {}", node, indeg.get(&id).unwrap());

                    for eid in fg.graph.edges_directed(id, Incoming) {
                        let edge = fg.graph.edge_weight(eid.id()).unwrap();
                        if !visited_edges.contains(&eid.id()) {
                            println!("  - Didn't visit edge {:?}", edge);
                        }
                    }
                }
            }
            assert!(false);
        }

        visited += topo_sort_order.len();

        // Condition chain where the previous stmt was inserted
        // This is because certain `Node` stmts depends on memport...
        for nidx in topo_sort_order {
            let node = fg.graph.node_weight(nidx).unwrap();
            match &node.nt {
                FirNodeType::Mux |
                FirNodeType::PrimOp2Expr(..) |
                FirNodeType::PrimOp1Expr(..) |
                FirNodeType::PrimOp1Expr1Int(..) |
                FirNodeType::PrimOp1Expr2Int(..) => {
                    let pconds = insert_op_stmts(fg, nidx, &emission_info, &array_addr_parents, &mut whentree);
                    emission_info.topdown.insert(nidx, pconds.clone());
                }
                FirNodeType::DontCare => {
                    let pconds = insert_invalidate_stmts(fg, nidx, &emission_info, &mut whentree);
                    emission_info.topdown.insert(nidx, pconds.clone());
                }
                FirNodeType::RegReset |
                FirNodeType::Reg      |
                FirNodeType::Wire     |
                FirNodeType::Inst(..) => {
                    let pconds = insert_def_topo_start_stmts(fg, nidx, &emission_info, &mut whentree);
                    emission_info.topdown.insert(nidx, pconds.clone());
                }
                FirNodeType::Memory(..) => {
                    let pconds = insert_def_firrtl3_mem_stmts(fg, nidx, &emission_info, &mut whentree);
                    emission_info.topdown.insert(nidx, pconds.clone());
                }
                _ => {
                    continue;
                }
            }
        }
    }

    if visited != fg.graph.node_count() {
        for id in fg.graph.node_indices() {
            if !vis_map.is_visited(&id) {
                let node = fg.graph.node_weight(id).unwrap();
                println!("Unvisited Node {:?}", node);
            }
        }
        assert!(
            visited == fg.graph.node_count(),
            "visited {} nodes out of {} nodes while topo sorting",
            visited,
            fg.graph.node_count());
    }

    // Insert assertions and printfs
    insert_printf_assertion_stmts(fg, &mut whentree);

    // Insert connection stmts
    insert_conn_stmts(fg, &mut whentree);

    // Export the WhenTree to stmts
    whentree.to_stmts(stmts);
}

fn insert_def_firrtl3_mem_stmts(
    fg: &FirGraph,
    id: NodeIndex,
    emission_info: &EmissionInfo,
    whentree: &mut WhenTree
) -> CondPath {
    let highest_path = find_highest_path(fg, id, &emission_info.topdown, whentree);
    let when_leaf = whentree.get_node_mut(
        &highest_path,
        Some(&highest_path.last().unwrap().prior))
            .unwrap();

    let node = fg.node_weight(id).unwrap();
    let mem_name = node.name.as_ref().unwrap().clone();
    match &node.nt {
        FirNodeType::Memory(depth, rlat, wlat, ports, ruw) => {
            let ttree = node.ttree.as_ref().unwrap();
            let first_port = ports.first().unwrap();
            let tpe = match first_port.as_ref() {
                MemoryPort::Read(name) |
                    MemoryPort::Write(name) => {
                    let port_ref = Reference::RefDot(Box::new(Reference::Ref(mem_name)), name.clone());
                    let reference = Reference::RefDot(Box::new(port_ref), Identifier::Name("data".to_string()));
                    ttree.view().unwrap().subtree_from_ref(&reference).unwrap().clone_ttree().to_type()
                }
                MemoryPort::ReadWrite(name) => {
                    let port_ref = Reference::RefDot(Box::new(Reference::Ref(mem_name)), name.clone());
                    let reference = Reference::RefDot(Box::new(port_ref), Identifier::Name("wdata".to_string()));
                    ttree.view().unwrap().subtree_from_ref(&reference).unwrap().clone_ttree().to_type()
                }
            };
            let name = node.name.as_ref().unwrap().clone();
            let stmt = Stmt::Memory(name, tpe, *depth, *rlat, *wlat, ports.clone(), ruw.clone(), Info::default());
            let pstmt = StmtWithPrior::new(stmt, None);
            when_leaf.stmts.push(pstmt);
        }
        _ => {
            unreachable!();
        }
    }
    highest_path
}
