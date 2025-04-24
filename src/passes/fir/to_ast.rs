use crate::common::graphviz::DefaultGraphVizCore;
use crate::ir::fir::{FirEdgeType, FirGraph, FirIR, FirNodeType};
use crate::ir::whentree::{CondPath, CondPathWithPrior, StmtWithPrior, WhenTree};
use chirrtl_parser::ast::*;
use std::collections::VecDeque;
use indexmap::IndexMap;
use indexmap::IndexSet;
use petgraph::{visit::EdgeRef, Direction::Incoming, Direction::Outgoing};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Undirected;
use petgraph::visit::VisitMap;
use petgraph::visit::Visitable;
use petgraph::prelude::Dfs;
use std::cmp::min as lower;

/// Converts the FirIR back to an AST
pub fn to_ast(fir: &FirIR) -> Circuit {
    let mut cms = CircuitModules::new();
    for hier in fir.hier.topo_order() {
        let fg = fir.graphs.get(hier.name()).unwrap();
        let cm = to_circuitmodule(hier.name(), fg);
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
    collect_stmts(fg, &mut stmts);
    Module::new(name.clone(), ports, stmts, Info::default())
}

/// Statements that defines a structural element
fn topo_start_node(fg: &FirGraph, id: NodeIndex) -> bool {
    let node = fg.graph.node_weight(id).unwrap();
    match node.nt {
        FirNodeType::Invalid  |
        FirNodeType::SMem(..) |
        FirNodeType::CMem     |
        FirNodeType::Input    |
        FirNodeType::Output   |
        FirNodeType::UIntLiteral(..)  |
        FirNodeType::SIntLiteral(..)  |
        FirNodeType::RegReset |
        FirNodeType::Reg      |
        FirNodeType::Wire     |
        FirNodeType::Inst(..) => {
            true
        }
        _ => {
            false
        }
    }
}

/// Finds all array nodes that are used as address inputs for a given array access
/// ref_expr: regfile_1[io.addr]
fn find_array_addr_chain(fg: &FirGraph, id: NodeIndex, ref_expr: &Expr) -> IndexSet<NodeIndex> {
    let mut array_parents: IndexSet<NodeIndex> = IndexSet::new();
    for peid in fg.graph.edges_directed(id, Incoming) {
        let pedge = fg.graph.edge_weight(peid.id()).unwrap();
        let ep = fg.graph.edge_endpoints(peid.id()).unwrap();

        // Only process ArrayAddr edges which represent array indexing operations
        if pedge.et == FirEdgeType::ArrayAddr {
            if let Expr::Reference(r) = &pedge.src {
                // If this edge's source matches our reference expression,
                // add it to dependencies
                if &pedge.src == ref_expr {
                    array_parents.insert(ep.0);
                }

                match r {
                    // For nested array accesses (e.g., regfile_1[io.addr] in our example)
                    // recursively find their dependencies
                    // - _par: regfile_1
                    // - leaf: io.addr
                    Reference::RefIdxExpr(_par, leaf) => {
                        if &pedge.src == ref_expr {
                            let mut p_arr = find_array_addr_chain(fg, ep.0, leaf.as_ref());
                            array_parents.append(&mut p_arr);
                        }
                    }
                    _ => {
                        continue;
                    }
                }
            }
        }
    }
    array_parents
}

/// Analyzes array dependencies in reference expressions and builds a dependency graph
/// For example, in `regfile_2[regfile_1[io.addr]]`, it:
/// 1. Processes the nested reference structure (RefIdxExpr)
/// 2. Finds all array accesses that must be evaluated first
/// 3. Updates array_addr_edges to track dependencies between arrays
/// 4. Maintains indeg counts for topological sorting
fn find_array_addr_chain_in_ref(
    fg: &FirGraph,
    src: NodeIndex,
    dst: NodeIndex,
    ref_expr: &Reference,
    whentree: &WhenTree,
    array_addr_edges: &mut IndexMap<NodeIndex, IndexSet<NodeIndex>>,
    indeg: &mut IndexMap<NodeIndex, u32>
) {
    match ref_expr {
        Reference::RefIdxExpr(x, y) => {
            // Handle array indexing expressions like regfile_2.a.bits[regfile_1[io.addr]]
            // - x: regfile_2.a.bits
            // - y: regfile_1[io.addr]

            // Find all arrays used in the index expression
            let chain = find_array_addr_chain(fg, src, y.as_ref());

            for addr_src in chain {
                // Handle special case for phi nodes in SSA form
                // where a node might feed back into itself.
                // In this case, if this node can be used as a starting node
                // during topological sort, don't increment the indeg count
                // as this phi node will not be traversed
                // - `connect a.b, my_array[a.b.c]`
                let phi_oedge_opt = fg.parent_with_type(addr_src, FirEdgeType::PhiOut);
                if phi_oedge_opt.is_some() {
                    let phi_oedge_id = phi_oedge_opt.unwrap();
                    let ep = fg.graph.edge_endpoints(phi_oedge_id).unwrap();
                    if ep.0 == dst && topo_start_node(fg, dst) {
                        continue;
                    }
                }

                // Build the dependency graph by recording that addr_src must be
                // evaluated before dst, and increment dst's indegree count
                if !array_addr_edges.contains_key(&addr_src) {
                    array_addr_edges.insert(addr_src, IndexSet::new());
                }
                let dsts = array_addr_edges.get_mut(&addr_src).unwrap();
                if !dsts.contains(&dst) {
                    dsts.insert(dst);
                    *indeg.get_mut(&dst).unwrap() += 1;
                }
            }
            // Recursively process the parent reference
            find_array_addr_chain_in_ref(fg, src, dst, x, whentree, array_addr_edges, indeg);
        }
        // Handle field access like 'io.addr'
        Reference::RefDot(a, _field) => {
            find_array_addr_chain_in_ref(fg, src, dst, a, whentree, array_addr_edges, indeg);
        }
        // Handle constant integer indexing like array[5]
        Reference::RefIdxInt(a, _idx) => {
            find_array_addr_chain_in_ref(fg, src, dst, a, whentree, array_addr_edges, indeg);
        }
        _ => {
        }
    }
}

#[derive(Debug, Default)]
struct EmissionInfo {
    pub topdown: IndexMap<NodeIndex, CondPath>,

    /// Node must have at most PrioritizedConds (ceiling in stmt list)
    pub bottomup: IndexMap<NodeIndex, CondPath>,
}

fn reconstruct_whentree(fg: &FirGraph) -> WhenTree {
    let mut cond_priority_pair = vec![];
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
        match &node.nt {
            FirNodeType::Phi(ppath_node) => {
                cond_priority_pair.push(&ppath_node.path);

                let eids = fg.graph.edges_directed(id, Incoming);
                for eid in eids {
                    let edge = fg.graph.edge_weight(eid.id()).unwrap();
                    if let FirEdgeType::PhiInput(ppath, _flipped) = &edge.et {
                        cond_priority_pair.push(&ppath.path);
                    }
                }
            }
            FirNodeType::ReadMemPort(ppath)  |
            FirNodeType::WriteMemPort(ppath) |
            FirNodeType::InferMemPort(ppath) => {
                cond_priority_pair.push(&ppath.path);
            }
            FirNodeType::Printf(.., ppath) |
            FirNodeType::Assert(.., ppath) => {
                cond_priority_pair.push(&ppath.path);
            }
            _ => {
                continue;
            }
        }
    }
    WhenTree::build_from_conditions(cond_priority_pair)
}

fn is_reginit(fg: &FirGraph, id: NodeIndex) -> bool {
    let node = fg.graph.node_weight(id).unwrap();
    node.nt == FirNodeType::RegReset
}

fn reverse_adjacency_list(
    graph: &IndexMap<NodeIndex, IndexSet<NodeIndex>>,
) -> IndexMap<NodeIndex, IndexSet<NodeIndex>> {
    let mut reversed = IndexMap::<NodeIndex, IndexSet<NodeIndex>>::new();

    for (src, dsts) in graph {
        for dst in dsts {
            reversed
                .entry(*dst)
                .or_insert_with(IndexSet::new)
                .insert(*src);
        }
    }

    reversed
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
        *indeg.get_mut(&dst).unwrap() += 1;
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

    // Check for cases where RegInit
    for eid in fg.graph.edge_indices() {
        let edge = fg.graph.edge_weight(eid).unwrap();
        if edge.et == FirEdgeType::InitValue {
            let ep = fg.graph.edge_endpoints(eid).unwrap();
            let node = fg.graph.node_weight(ep.1).unwrap();

            assert!(node.nt == FirNodeType::RegReset);
            *indeg.get_mut(&ep.1).unwrap() = 3;
        }
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
                if !is_reginit(fg, nx) || *indeg.get(&nx).unwrap() == 0 {
                    q.push_back(nx);
                }
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
                    if is_reginit(fg, cidx) {
                        continue;
                    }
                    *indeg.get_mut(&cidx).unwrap() -= 1;
                    if *indeg.get(&cidx).unwrap() == 0 {
                        q.push_back(cidx);
                    }
                }
            }

            if invalidate_edges.contains_key(&nidx) {
                let inv_childs = invalidate_edges.get(&nidx).unwrap();
                for &cidx in inv_childs {
                    if is_reginit(fg, cidx) {
                        continue;
                    }
                    *indeg.get_mut(&cidx).unwrap() -= 1;
                    if *indeg.get(&cidx).unwrap() == 0 {
                        q.push_back(cidx);
                    }
                }
            }

            let cedges = fg.graph.edges_directed(nidx, Outgoing);
            for cedge in cedges {
                let ep = fg.graph.edge_endpoints(cedge.id()).unwrap();
                let edge = fg.graph.edge_weight(cedge.id()).unwrap();
                 if !topo_vis_map.is_visited(&ep.1) && is_reginit(fg, ep.1) &&
                    (edge.et == FirEdgeType::Clock ||
                     edge.et == FirEdgeType::Reset ||
                     edge.et == FirEdgeType::InitValue)
                {
                    *indeg.get_mut(&ep.1).unwrap() -= 1;
                    if *indeg.get(&ep.1).unwrap() == 0 {
                        q.push_back(ep.1);
                    }
                } else if !topo_vis_map.is_visited(&ep.1) && !topo_start_node(&fg, ep.1) {
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
                    println!("Visited during DFS Node {:?}, indeg {}", node, indeg.get(&id).unwrap());

                    for eid in fg.graph.edges_directed(id, Incoming) {
                        let edge = fg.graph.edge_weight(eid.id()).unwrap();
                        if !visited_edges.contains(&eid.id()) {
                            println!("didn't visit edge {:?}", edge);
                        }
                    }
                }
            }
            assert!(false);
        }

        visited += topo_sort_order.len();

        fill_bottom_up_emission_info(fg, &topo_sort_order, &whentree, &mut emission_info);

        // Condition chain where the previous stmt was inserted
        // This is because certain `Node` stmts depends on memport...
        for nidx in topo_sort_order {
            let node = fg.graph.node_weight(nidx).unwrap();
            match &node.nt {
                FirNodeType::ReadMemPort(..) |
                FirNodeType::WriteMemPort(..) |
                FirNodeType::InferMemPort(..) => {
                    let pconds = insert_memport_stmts(fg, nidx, &mut whentree);
                    emission_info.topdown.insert(nidx, pconds.clone());
                }
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
                FirNodeType::SMem(..) |
                FirNodeType::CMem => {
                    let pconds = insert_def_mem_stmts(fg, nidx, &emission_info, &mut whentree);
                    emission_info.topdown.insert(nidx, pconds.clone());
                }
                FirNodeType::RegReset |
                FirNodeType::Reg      |
                FirNodeType::Wire     |
                FirNodeType::Inst(..) => {
                    let pconds = insert_def_topo_start_stmts(fg, nidx, &emission_info, &mut whentree);
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

/// Looks at the parent nodes (drivers) and their places in the stmt hierarchy.
/// This node must be located lower than any of its parent node
fn find_highest_path(
    fg: &FirGraph,
    id: NodeIndex,
    node_to_path: &IndexMap<NodeIndex, CondPath>,
    whentree: &WhenTree
) -> CondPath {
    let parents = fg.graph.neighbors_directed(id, Incoming);
    let mut highest_path = whentree.get_top_path();
    for pid in parents {
        if node_to_path.contains_key(&pid) {
            let pconds = node_to_path.get(&pid).unwrap();
            highest_path = lower(highest_path, pconds.clone());
        }
    }
    highest_path
}

fn insert_def_mem_stmts(
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
    match &node.nt {
        FirNodeType::SMem(ruw_opt) => {
            let tpe = node.ttree.as_ref().unwrap().to_type();

            let name = node.name.as_ref().unwrap().clone();
            let smem = ChirrtlMemory::SMem(name, tpe, ruw_opt.clone(), Info::default());
            let mem = Stmt::ChirrtlMemory(smem);
            let pstmt = StmtWithPrior::new(mem, None);
            when_leaf.stmts.push(pstmt);
        }
        FirNodeType::CMem => {
            let tpe = node.ttree.as_ref().unwrap().to_type();

            let name = node.name.as_ref().unwrap().clone();
            let cmem = ChirrtlMemory::CMem(name, tpe, Info::default());
            let mem = Stmt::ChirrtlMemory(cmem);
            let pstmt = StmtWithPrior::new(mem, None);
            when_leaf.stmts.push(pstmt);
        }
        _ => {
            unreachable!();
        }
    }
    highest_path
}

fn insert_def_topo_start_stmts(
    fg: &FirGraph,
    id: NodeIndex,
    emission_info: &EmissionInfo,
    whentree: &mut WhenTree
) -> CondPath {
    let node = fg.node_weight(id).unwrap();
    let mut highest_path = whentree.get_top_path();

    match &node.nt {
        FirNodeType::RegReset => {
            let clk_eid  = fg.parent_with_type(id, FirEdgeType::Clock).unwrap();
            let rst_eid = fg.parent_with_type(id, FirEdgeType::Reset).unwrap();
            let init_eid = fg.parent_with_type(id, FirEdgeType::InitValue).unwrap();

            let clk_src = fg.graph.edge_endpoints(clk_eid).unwrap().0;
            let rst_src = fg.graph.edge_endpoints(rst_eid).unwrap().0;
            let init_src = fg.graph.edge_endpoints(init_eid).unwrap().0;

            if emission_info.topdown.contains_key(&clk_src) {
                let pconds = emission_info.topdown.get(&clk_src).unwrap();
                highest_path = lower(highest_path, pconds.clone());
            }
            if emission_info.topdown.contains_key(&rst_src) {
                let pconds = emission_info.topdown.get(&rst_src).unwrap();
                highest_path = lower(highest_path, pconds.clone());
            }
            if emission_info.topdown.contains_key(&init_src) {
                let pconds = emission_info.topdown.get(&init_src).unwrap();
                highest_path = lower(highest_path, pconds.clone());
            }
        }
        _ => {}
    }

    if let Some(peid) = fg.parent_with_type(id, FirEdgeType::PhiOut) {
        let phi_id = fg.graph.edge_endpoints(peid).unwrap().0;
        let phi = fg.graph.node_weight(phi_id).unwrap();
        if let FirNodeType::Phi(pcond) = &phi.nt {
            if pcond != &CondPathWithPrior::default() {
                highest_path = lower(highest_path, pcond.path.clone());
            }
        } else {
            unreachable!();
        }
    }

    let when_leaf = whentree.get_node_mut(
        &highest_path,
        Some(&highest_path.last().unwrap().prior))
            .unwrap();

    match &node.nt {
        FirNodeType::RegReset => {
            let tpe = node.ttree.as_ref().unwrap().to_type();

            let clk_eid = fg.parent_with_type(id, FirEdgeType::Clock).unwrap();
            let rst_eid = fg.parent_with_type(id, FirEdgeType::Reset).unwrap();
            let init_eid = fg.parent_with_type(id, FirEdgeType::InitValue).unwrap();

            let clk  = fg.graph.edge_weight(clk_eid).unwrap().src.clone();
            let rst  = fg.graph.edge_weight(rst_eid).unwrap().src.clone();
            let init = fg.graph.edge_weight(init_eid).unwrap().src.clone();

            let name = node.name.as_ref().unwrap().clone();
            let reg_init = Stmt::RegReset(name, tpe, clk, rst, init, Info::default());
            let pstmt = StmtWithPrior::new(reg_init, None);
            when_leaf.stmts.push(pstmt);
        }
        FirNodeType::Reg => {
            let tpe = node.ttree.as_ref().unwrap().to_type();

            let clk_eid = fg.parent_with_type(id, FirEdgeType::Clock).unwrap();
            let clk = fg.graph.edge_weight(clk_eid).unwrap().src.clone();

            let name = node.name.as_ref().unwrap().clone();
            let reg = Stmt::Reg(name, tpe, clk, Info::default());
            let pstmt = StmtWithPrior::new(reg, None);
            when_leaf.stmts.push(pstmt);
        }
        FirNodeType::Wire => {
            let node = fg.node_weight(id).unwrap();
            let tpe = node.ttree.as_ref().unwrap().to_type();
            let name = node.name.as_ref().unwrap().clone();
            let stmt = Stmt::Wire(name, tpe, Info::default());
            let pstmt = StmtWithPrior::new(stmt, None);
            when_leaf.stmts.push(pstmt);
        }
        FirNodeType::Inst(..) => {
            let node = fg.node_weight(id).unwrap();
            let stmt = if let FirNodeType::Inst(module) = &node.nt {
                let name = node.name.as_ref().unwrap().clone();
                Stmt::Inst(name, module.clone(), Info::default())
            } else {
                unreachable!()
            };
            let pstmt = StmtWithPrior::new(stmt, None);
            when_leaf.stmts.push(pstmt);
        }
        _ => {
            unreachable!();
        }
    }
    highest_path
}

/// Collect invalidate stmts
fn insert_invalidate_stmts(
    fg: &FirGraph,
    id: NodeIndex,
    emission_info: &EmissionInfo,
    whentree: &mut WhenTree
) -> CondPath {
    let mut highest_path = find_highest_path(fg, id, &emission_info.topdown, whentree);
    let childs = fg.graph.neighbors_directed(id, Outgoing);
    for cid in childs {
        let child = fg.graph.node_weight(cid).unwrap();
        match &child.nt {
            FirNodeType::Phi(pcond) => {
                if pcond != &CondPathWithPrior::default() {
                    highest_path = lower(highest_path, pcond.path.clone());
                }
            }
            _ => { }
        }
    }

    let when_leaf = whentree.get_node_mut(
        &highest_path,
        Some(&highest_path.last().unwrap().prior))
            .unwrap();

    let edges = fg.graph.edges_directed(id, Outgoing);
    for eref in edges {
        let eid = eref.id();
        let edge = fg.graph.edge_weight(eid).unwrap();
        if edge.dst.is_none() {
            continue;
        }

        let lhs = Expr::Reference(edge.dst.as_ref().unwrap().clone());
        if edge.et == FirEdgeType::DontCare {
            let stmt = Stmt::Invalidate(lhs, Info::default());
            let pstmt = StmtWithPrior::new(stmt, None);
            when_leaf.stmts.push(pstmt);
        };
    }

    highest_path
}


/// Add memory port definition statements
fn insert_memport_stmts(
    fg: &FirGraph,
    id: NodeIndex,
    whentree: &mut WhenTree
) -> CondPath {
    let node = fg.graph.node_weight(id).unwrap();
    match &node.nt {
        FirNodeType::ReadMemPort(ppath)      |
            FirNodeType::WriteMemPort(ppath) |
            FirNodeType::InferMemPort(ppath) => {
                let clk_eid  = fg.parent_with_type(id, FirEdgeType::Clock).unwrap();
                let addr_eid = fg.parent_with_type(id, FirEdgeType::MemPortAddr).unwrap();
                let mem_eid = fg.parent_with_type(id, FirEdgeType::MemPortEdge).unwrap();

                let mem_ep = fg.graph.edge_endpoints(mem_eid).unwrap();
                let mem_node = fg.graph.node_weight(mem_ep.0).unwrap();
                let mem_name = mem_node.name.as_ref().unwrap().clone();

                let mem = &fg.graph.edge_weight(mem_eid).unwrap().src;
                let clk = &fg.graph.edge_weight(clk_eid).unwrap().src;
                let addr = fg.graph.edge_weight(addr_eid).unwrap().src.clone();

                let port_stmt = if let Expr::Reference(clk_ref) = clk {
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
                let when_leaf = whentree.get_node_mut(
                    &ppath.path,
                    Some(&ppath.path.last().unwrap().prior)).unwrap();
                let pstmt = StmtWithPrior::new(port_stmt, None);
                when_leaf.stmts.push(pstmt);
                ppath.path.clone()
        }
        _ => {
            unreachable!();
        }
    }
}

/// Collect the connect and mport stmts and insert them into the WhenTree
fn insert_op_stmts(
    fg: &FirGraph,
    id: NodeIndex,
    emission_info: &EmissionInfo,
    array_addr_parents: &IndexMap<NodeIndex, IndexSet<NodeIndex>>,
    whentree: &mut WhenTree
) -> CondPath {
    let node = fg.graph.node_weight(id).unwrap();
    let mut highest_path = find_highest_path(fg, id, &emission_info.topdown, &whentree);

    if array_addr_parents.contains_key(&id) {
        let parents = array_addr_parents.get(&id).unwrap();
        for pid in parents {
            if emission_info.topdown.contains_key(pid) {
                highest_path = lower(highest_path, emission_info.topdown.get(pid).unwrap().clone());
            }
        }
    }

    let bu_pconds = emission_info.bottomup.get(&id);
    if bu_pconds.is_some() {
        let bu_pconds = bu_pconds.unwrap();
        if *bu_pconds > highest_path {
            println!("bu_pconds {:?}", bu_pconds);
            println!("highest_path {:?}", highest_path);
            println!("node {:?}", fg.graph.node_weight(id).unwrap());
            whentree.print_tree();
            assert!(false);
        }

        if let Some(x) = whentree.find_middle_ground(&highest_path, bu_pconds) {
            highest_path = x;
        }
    }

    let when_leaf = whentree.get_node_mut(
        &highest_path,
        Some(&highest_path.last().unwrap().prior))
            .unwrap();

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
            let pstmt = StmtWithPrior::new(stmt, None);
            when_leaf.stmts.push(pstmt);
        }
        FirNodeType::PrimOp2Expr(op) => {
            let op0_eid = fg.parent_with_type(id, FirEdgeType::Operand0).unwrap();
            let op1_eid = fg.parent_with_type(id, FirEdgeType::Operand1).unwrap();

            let op0 = fg.graph.edge_weight(op0_eid).unwrap().src.clone();
            let op1 = fg.graph.edge_weight(op1_eid).unwrap().src.clone();
            let primop = Expr::PrimOp2Expr(op.clone(), Box::new(op0), Box::new(op1));

            let name = node.name.as_ref().unwrap().clone();
            let stmt = Stmt::Node(name, primop, Info::default());
            let pstmt = StmtWithPrior::new(stmt, None);
            when_leaf.stmts.push(pstmt);
        }
        FirNodeType::PrimOp1Expr(op) => {
            if let Some(name) = node.name.as_ref() {
                let op0_eid = fg.parent_with_type(id, FirEdgeType::Operand0).unwrap();
                let op0 = fg.graph.edge_weight(op0_eid).unwrap().src.clone();
                let primop = Expr::PrimOp1Expr(op.clone(), Box::new(op0));

                let stmt = Stmt::Node(name.clone(), primop, Info::default());
                let pstmt = StmtWithPrior::new(stmt, None);
                when_leaf.stmts.push(pstmt);
            }
        }
        FirNodeType::PrimOp1Expr1Int(op, x) => {
            let op0_eid = fg.parent_with_type(id, FirEdgeType::Operand0).unwrap();
            let op0 = fg.graph.edge_weight(op0_eid).unwrap().src.clone();
            let primop = Expr::PrimOp1Expr1Int(op.clone(), Box::new(op0), x.clone());

            let name = node.name.as_ref().unwrap().clone();
            let stmt = Stmt::Node(name, primop, Info::default());
            let pstmt = StmtWithPrior::new(stmt, None);
            when_leaf.stmts.push(pstmt);
        }
        FirNodeType::PrimOp1Expr2Int(op, x, y) => {
            let op0_eid = fg.parent_with_type(id, FirEdgeType::Operand0).unwrap();
            let op0 = fg.graph.edge_weight(op0_eid).unwrap().src.clone();
            let primop = Expr::PrimOp1Expr2Int(op.clone(), Box::new(op0), x.clone(), y.clone());

            let name = node.name.as_ref().unwrap().clone();
            let stmt = Stmt::Node(name, primop, Info::default());
            let pstmt = StmtWithPrior::new(stmt, None);
            when_leaf.stmts.push(pstmt);
        }
        _ => {
            unreachable!();
        }
    }
    highest_path
}

fn fill_bottom_up_emission_info(
    fg: &FirGraph,
    topo_sort_order: &Vec<NodeIndex>,
    whentree: &WhenTree,
    emission_info: &mut EmissionInfo
) {
    for &id in topo_sort_order.iter().rev() {
        let cedges = fg.graph.edges_directed(id, Outgoing);
        let mut conds: Vec<CondPath> = vec![];
        for ceid in cedges {
            let edge = fg.graph.edge_weight(ceid.id()).unwrap();
            let cid = fg.graph.edge_endpoints(ceid.id()).unwrap().1;
            if edge.dst.is_some() {
                match &edge.et {
                    FirEdgeType::DontCare => {
                        continue;
                    }
                    FirEdgeType::PhiInput(pconds, _flipped) => {
                        conds.push(pconds.path.clone());
                    }
                    _ => {
                        conds.push(CondPath::bottom());
                    }
                }
            } else if emission_info.bottomup.contains_key(&cid) {
                conds.push(emission_info.bottomup.get(&cid).unwrap().clone());
            } else {
                match edge.et {
                    FirEdgeType::PhiSel => {
                        let parents = fg.graph.edges_directed(cid, Incoming);
                        for peid in parents {
                            let pedge = fg.graph.edge_weight(peid.id()).unwrap();
                            if let FirEdgeType::PhiInput(pconds, _flipped) = &pedge.et {
                                if pconds.path.collect_sels().contains(&edge.src) {
                                    let x = pconds.path.cond_path(&edge.src);
                                    conds.push(x);
                                }
                            }
                        }
                    }
                    _ => { }
                }
            }
        }
        let pcond_constraint = whentree.bottom_up_priority_constraint(&conds);
        if pcond_constraint.is_some() {
            emission_info.bottomup.insert(id, pcond_constraint.unwrap());
        } else {
        }
    }
}

fn insert_printf_assertion_stmts(fg: &FirGraph, whentree: &mut WhenTree) {
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
        match &node.nt {
            FirNodeType::Printf(stmt, pcond) |
            FirNodeType::Assert(stmt, pcond) => {
                let leaf = whentree.get_node_mut(&pcond.path, None).unwrap();
                let pstmt = StmtWithPrior::new(stmt.clone(), None);
                leaf.stmts.push(pstmt);
            }
            _ => {
                continue;
            }
        }
    }
}

/// Insert connection stmts to the appropriate position in the whentree
fn insert_conn_stmts(
    fg: &FirGraph,
    whentree: &mut WhenTree
) {
    // Collect stmts according to their position in the whentree
    let mut ordered_stmts: IndexMap<&CondPath, Vec<StmtWithPrior>> = IndexMap::new();
    for eid in fg.graph.edge_indices() {
        let edge = fg.graph.edge_weight(eid).unwrap();
        if edge.dst.is_none() {
            continue;
        }

        let lhs = Expr::Reference(edge.dst.as_ref().unwrap().clone());
        let rhs = edge.src.clone();
        match &edge.et {
            FirEdgeType::DontCare => {
                continue;
            }
            FirEdgeType::PhiInput(ppath, flipped) => {
                if !ordered_stmts.contains_key(&ppath.path) {
                    ordered_stmts.insert(&ppath.path, vec![]);
                }
                let stmt = if *flipped {
                    Stmt::Connect(rhs, lhs, Info::default())
                } else {
                    Stmt::Connect(lhs, rhs, Info::default())
                };
                let pstmt = StmtWithPrior::new(stmt, Some(ppath.prior));
                ordered_stmts.get_mut(&ppath.path).unwrap().push(pstmt);
            }
            _ => {
                let leaf = whentree.get_node_mut(&CondPath::bottom(), None).unwrap();
                let stmt = Stmt::Connect(lhs, rhs, Info::default());
                let pstmt = StmtWithPrior::new(stmt, None);
                leaf.stmts.push(pstmt);
            }
        }
    }

    // Sort the connection stmts accordint to their priority in the whentree
    for stmts in ordered_stmts.values_mut() {
        stmts.sort_by(|a, b| b.cmp(&a));
    }

    // Insert the sorted stmts to the correct position
    for (conds, pstmts) in ordered_stmts {
        for pstmt in pstmts {
            let whentree_prior = conds.last().unwrap().prior;
            let leaf = whentree.get_node_mut(conds, Some(&whentree_prior)).unwrap();
            leaf.stmts.push(pstmt);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::common::export_circuit;
    use crate::ir::whentree::*;
    use crate::passes::ast::print::Printer;
    use crate::common::RippleIRErr;
    use crate::passes::runner::run_fir_passes_from_circuit;
    use chirrtl_parser::parse_circuit;
    use test_case::test_case;
    use indexmap::IndexMap;

    fn check_whentree_equivalence(ir: &FirIR, circuit: &Circuit) -> Result<(), RippleIRErr> {
        let mut ast_whentrees: IndexMap<&Identifier, WhenTree> = IndexMap::new();
        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    let whentree = WhenTree::build_from_stmts(&m.stmts);
                    ast_whentrees.insert(&m.name, whentree);
                }
                CircuitModule::ExtModule(_) => {
                    continue;
                }
            }
        }

        for (name, fg) in ir.graphs.iter() {
            if ast_whentrees.contains_key(name) {
                let ast_whentree = ast_whentrees.get(name).unwrap();
                let ast_leaves = ast_whentree.leaf_to_paths();

                let fir_whentree = reconstruct_whentree(fg);
                let fir_leaves = fir_whentree.leaf_to_paths();

                if name == &Identifier::Name("Directory".to_string()) {
                    fir_whentree.print_tree();
                }

                for (fnode, fconds) in fir_leaves {
                    if !ast_leaves.contains_key(fnode) {
                        assert_eq!(fnode.cond, Condition::Root);
                        assert!(
                            (fnode.prior == PhiPrior::top()) ||
                            (fnode.prior == PhiPrior::bottom()));
                    } else {
                        let aconds = ast_leaves.get(fnode).unwrap();
                        assert_eq!(aconds, &fconds);
                    }
                }
            }
        }
        Ok(())
    }

    #[test_case("GCD" ; "GCD")]
    #[test_case("NestedWhen" ; "NestedWhen")]
    #[test_case("NestedIndex" ; "NestedIndex")]
    #[test_case("LCS1" ; "LCS1")]
    #[test_case("LCS2" ; "LCS2")]
    #[test_case("LCS3" ; "LCS3")]
    #[test_case("LCS4" ; "LCS4")]
    #[test_case("LCS5" ; "LCS5")]
    #[test_case("LCS6" ; "LCS6")]
    #[test_case("LCS7" ; "LCS7")]
    #[test_case("LCS8" ; "LCS8")]
    #[test_case("BitSel1" ; "BitSel1")]
    #[test_case("BitSel2" ; "BitSel2")]
    #[test_case("RegInit" ; "RegInit")]
    #[test_case("RegInitWire" ; "RegInitWire")]
    #[test_case("SinglePortSRAM" ; "SinglePortSRAM")]
    #[test_case("OneReadOneWritePortSRAM" ; "OneReadOneWritePortSRAM")]
    #[test_case("OneReadOneReadWritePortSRAM" ; "OneReadOneReadWritePortSRAM")]
    #[test_case("MSHR" ; "MSHR")]
    #[test_case("EmptyAggregate" ; "EmptyAggregate")]
    #[test_case("TLFIFOFixer" ; "TLFIFOFixer")]
    #[test_case("TLBundleQueue" ; "TLBundleQueue")]
    #[test_case("ListBuffer" ; "ListBuffer")]
    #[test_case("Atomics" ; "Atomics")]
    #[test_case("PhitArbiter" ; "PhitArbiter")]
    #[test_case("TLMonitor" ; "TLMonitor")]
    #[test_case("TLBusBypassBar" ; "TLBusBypassBar")]
    #[test_case("WireRegInsideWhen" ; "WireRegInsideWhen")]
    #[test_case("MultiWhen" ; "MultiWhen")]
    #[test_case("DCacheDataArray" ; "DCacheDataArray")]
    #[test_case("Hierarchy" ; "Hierarchy")]
    #[test_case("Cache" ; "Cache")]
    #[test_case("TLXbar_pbus" ; "TLXbar_pbus")]
    #[test_case("BTBBranchPredictorBank" ; "BTBBranchPredictorBank")]
    #[test_case("chipyard.harness.TestHarness.RocketConfig" ; "Rocket")]
    #[test_case("chipyard.harness.TestHarness.LargeBoomV3Config" ; "Boom")]
    fn run(name: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(format!("./test-inputs/{}.fir", name))?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let ir = run_fir_passes_from_circuit(&circuit)?;
        check_whentree_equivalence(&ir, &circuit)?;

        let circuit_reconstruct = to_ast(&ir);

        let firrtl = format!("./test-outputs/{}.fir", name);

        let mut printer = Printer::new();
        let circuit_str = printer.print_circuit(&circuit_reconstruct);
        std::fs::write(&firrtl, circuit_str)?;
        export_circuit(&firrtl, &format!("test-outputs/{}/verilog", name))?;

        Ok(())
    }
}
