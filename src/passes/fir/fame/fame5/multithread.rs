use crate::passes::fir::fame::fame5::memory::*;
use crate::passes::fir::fame::fame5::register::*;
use crate::passes::fir::fame::fame5::*;
use crate::ir::typetree::typetree::TypeTree;
use crate::ir::typetree::tnode::GroundType;
use crate::passes::fir::fame::fame5::top::fame5_name;
use petgraph::graph::NodeIndex;
use rusty_firrtl::Identifier;

pub fn multithread_module(
    fg: &FirGraph,
    nthreads: u32,
    host_clock: &Identifier,
    host_reset: &Identifier,
    blackboxes: &HashSet<Identifier>
) -> FirGraph {
    let mut fame5 = fg.clone();

    let mut reg_or_mem_ids: Vec<NodeIndex> = fame5.graph.node_indices().filter(|id| {
        let node = fame5.graph.node_weight(*id).unwrap();
        match node.nt {
            FirNodeType::Reg |
                FirNodeType::RegReset |
                FirNodeType::Memory(..) => {
                    true
                }
            _ => {
                false
            }
        }
    }).collect();
    reg_or_mem_ids.sort();

    add_host_clock_and_reset(&mut fame5, host_clock, host_reset);

    let thread_idx_id = add_thread_idx_reg(&mut fame5, nthreads, host_clock, host_reset);
    add_thread_idx_update(&mut fame5, thread_idx_id, nthreads, log2_ceil(nthreads));

    let host_clock_id = find_host_clock_or_reset_id(&fame5, host_clock);

    for &id in reg_or_mem_ids.iter().rev() {
        let node = fame5.graph.node_weight(id).unwrap().clone();
        match node.nt {
            FirNodeType::Reg |
                FirNodeType::RegReset => {
                    duplicate_register(&mut fame5, nthreads, thread_idx_id, id);
                }
            FirNodeType::Memory(_, rlat, wlat, _, _) => {
                assert!(rlat <= 1);
                assert!(wlat == 1);
                duplicate_memory(&mut fame5, nthreads, thread_idx_id, id, host_clock_id, host_clock);
            }
            _ => {
                unreachable!()
            }
        }
        fame5.graph.remove_node(id);
    }

    // Find all instance nodes and update them
    let inst_ids: Vec<NodeIndex> = fame5.graph.node_indices().filter(|id| {
        let node = fame5.graph.node_weight(*id).unwrap();
        matches!(node.nt, FirNodeType::Inst(_))
    }).collect();

    for &inst_id in inst_ids.iter() {
        let node = fame5.graph.node_weight(inst_id).unwrap();
        if let FirNodeType::Inst(module_name) = &node.nt {
            if !blackboxes.contains(module_name) {
                let mut updated_node = node.clone();
                let fame5_module_name = Identifier::Name(fame5_name(module_name));
                updated_node.nt = FirNodeType::Inst(fame5_module_name);
                fame5.graph[inst_id] = updated_node;

                connect_to_host_clock_and_reset(&mut fame5, host_clock, host_reset, inst_id);
            }
        }
    }
    fame5
}

/// Add host clock and reset nodes if they don't exist
fn add_host_clock_and_reset(fame5: &mut FirGraph, host_clock: &Identifier, host_reset: &Identifier) {
    let mut host_clock_exists = false;
    let mut host_reset_exists = false;

    // Check if host_clock and host_reset nodes already exist in a single pass
    for id in fame5.graph.node_indices() {
        let node = fame5.graph.node_weight(id).unwrap();
        if let Some(name) = &node.name {
            if name == host_clock {
                host_clock_exists = true;
            }
            if name == host_reset {
                host_reset_exists = true;
            }
            // Early exit if both nodes are found
            if host_clock_exists && host_reset_exists {
                break;
            }
        }
    }

    // Add host_clock node if it doesn't exist
    if !host_clock_exists {
        let clock_node = FirNode::new(
            Some(host_clock.clone()),
            FirNodeType::Input,
            Some(TypeTree::build_from_ground_type(GroundType::Clock))
        );
        fame5.graph.add_node(clock_node);
    }

    // Add host_reset node if it doesn't exist
    if !host_reset_exists {
        let reset_node = FirNode::new(
            Some(host_reset.clone()),
            FirNodeType::Input,
            Some(TypeTree::build_from_ground_type(GroundType::Reset))
        );
        fame5.graph.add_node(reset_node);
    }
}

/// Connect host clock and reset to an instance node
fn connect_to_host_clock_and_reset(
    fg: &mut FirGraph,
    host_clock: &Identifier,
    host_reset: &Identifier,
    dst_id: NodeIndex
) {
    let host_clock_id = find_host_clock_or_reset_id(fg, host_clock);
    let host_reset_id = find_host_clock_or_reset_id(fg, host_reset);
    let dst = fg.graph.node_weight(dst_id).unwrap();
    let name = dst.name.as_ref().unwrap().clone();

    let hc_edge = FirEdge::new(
        Expr::Reference(Reference::Ref(host_clock.clone())),
        Some(Reference::RefDot(Box::new(Reference::Ref(name.clone())), host_clock.clone())),
        FirEdgeType::Clock);
    fg.graph.add_edge(host_clock_id, dst_id, hc_edge);

    let hr_edge = FirEdge::new(
        Expr::Reference(Reference::Ref(host_reset.clone())),
        Some(Reference::RefDot(Box::new(Reference::Ref(name.clone())), host_reset.clone())),
        FirEdgeType::Reset);
    fg.graph.add_edge(host_reset_id, dst_id, hr_edge);
}
