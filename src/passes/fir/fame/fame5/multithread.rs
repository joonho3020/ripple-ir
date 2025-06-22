use std::mem;

use crate::passes::fir::fame::fame5::*;
use crate::passes::fir::from_ast::memory_type_from_ports;
use crate::ir::typetree::typetree::TypeTree;
use crate::ir::typetree::tnode::{TypeDirection, GroundType};
use crate::passes::fir::fame::fame5::top::fame5_name;
use petgraph::graph::NodeIndex;
use rusty_firrtl::{Identifier, MemoryPort, ChirrtlMemoryReadUnderWrite, Width};
use petgraph::Direction;
use petgraph::graph::EdgeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction::{Incoming, Outgoing};

fn find_edge_with_type(fg: &FirGraph, id: NodeIndex, et: FirEdgeType, dir: Direction) -> Option<EdgeIndex> {
    for eid in fg.graph.edges_directed(id, dir) {
        let edge = fg.graph.edge_weight(eid.id()).unwrap();
        if edge.et == et {
            return Some(eid.id());
        }
    }
    None
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

pub fn multithread_module(
    fg: &FirGraph,
    nthreads: u32,
    host_clock: &Identifier,
    host_reset: &Identifier,
    blackboxes: &HashSet<Identifier>
) -> FirGraph {
    let mut fame5 = fg.clone();

    let mut reg_ids: Vec<NodeIndex> = fame5.graph.node_indices().filter(|id| {
        let node = fame5.graph.node_weight(*id).unwrap();
        node.nt == FirNodeType::Reg ||
            node.nt == FirNodeType::RegReset
    }).collect();
    reg_ids.sort();


    add_host_clock_and_reset(&mut fame5, host_clock, host_reset);

    let thread_idx_id = add_thread_idx_reg(&mut fame5, nthreads, host_clock, host_reset);
    let thread_idx_name = fame5.graph.node_weight(thread_idx_id).unwrap().name.as_ref().unwrap().clone();
    add_thread_idx_update(&mut fame5, thread_idx_id, nthreads, log2_ceil(nthreads));

    for &reg_id in reg_ids.iter().rev() {
        let rport_name = Identifier::Name("rd".to_string());
        let rport = MemoryPort::Read(rport_name.clone());

        let wport_name = Identifier::Name("wr".to_string());
        let wport = MemoryPort::Write(wport_name.clone());
        let ports = vec![
            Box::new(rport),
            Box::new(wport),
        ];

        let node = fame5.graph.node_weight(reg_id).unwrap().clone();
        let reg_name = node.name.as_ref().unwrap().clone();
        let comb_mem_type = memory_type_from_ports(&ports, nthreads, &node.ttree.as_ref().unwrap().to_type());
        let comb_mem = FirNodeType::Memory(nthreads, 0, 1, ports, ChirrtlMemoryReadUnderWrite::Undefined);

        // Make register into array
        let reg_threaded = FirNode::new(
            Some(reg_name.clone()),
            comb_mem,
            Some(TypeTree::build_from_type(&comb_mem_type, TypeDirection::Outgoing)));

        let reg_threaded_id = fame5.graph.add_node(reg_threaded);
        let rport_ref = Reference::RefDot(
            Box::new(Reference::Ref(reg_name.clone())),
            rport_name.clone());

        let wport_ref = Reference::RefDot(
            Box::new(Reference::Ref(reg_name.clone())),
            wport_name.clone());

        let uint1 = FirNode::new(None, FirNodeType::UIntLiteral(Width(1), Int::from(1)), None);

        let clock_eid = find_edge_with_type(&fame5, reg_id, FirEdgeType::Clock, Incoming).unwrap();
        let clock_id = fame5.graph.edge_endpoints(clock_eid).unwrap().0;
        let clock_edge = fame5.graph.edge_weight(clock_eid).unwrap().clone();

        // Add clock to read port
        let mut rport_clock_edge = clock_edge.clone();
        rport_clock_edge.dst = Some(Reference::RefDot(Box::new(rport_ref.clone()), Identifier::Name("clk".to_string())));
        fame5.graph.add_edge(clock_id, reg_threaded_id, rport_clock_edge);

        // Add enable to read port
        let rport_en_id = fame5.graph.add_node(uint1.clone());
        let rport_en_edge = FirEdge::new(
            Expr::UIntInit(Width(1), Int::from(1)),
            Some(Reference::RefDot(Box::new(rport_ref.clone()), Identifier::Name("en".to_string()))),
            FirEdgeType::Wire);
        fame5.graph.add_edge(rport_en_id, reg_threaded_id, rport_en_edge);

        // Add addr to read port
        let rport_addr_edge = FirEdge::new(
            Expr::Reference(Reference::Ref(thread_idx_name.clone())),
            Some(Reference::RefDot(Box::new(rport_ref.clone()), Identifier::Name("addr".to_string()))),
            FirEdgeType::Wire);
        fame5.graph.add_edge(thread_idx_id, reg_threaded_id, rport_addr_edge);

        // Add clock to write port
        let mut wport_clock_edge = clock_edge.clone();
        wport_clock_edge.dst = Some(Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("clk".to_string())));
        fame5.graph.add_edge(clock_id, reg_threaded_id, wport_clock_edge);

        // Add mask to write port
        let wport_mask_id = fame5.graph.add_node(uint1.clone());
        let wport_mask_edge = FirEdge::new(
            Expr::UIntInit(Width(1), Int::from(1)),
            Some(Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("mask".to_string()))),
            FirEdgeType::Wire);
        fame5.graph.add_edge(wport_mask_id, reg_threaded_id, wport_mask_edge);

        // Add addr to write port
        let wport_addr_edge = FirEdge::new(
            Expr::Reference(Reference::Ref(thread_idx_name.clone())),
            Some(Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("addr".to_string()))),
            FirEdgeType::Wire);
        fame5.graph.add_edge(thread_idx_id, reg_threaded_id, wport_addr_edge);

        // Add register sink edges
        let mut sink_edges: Vec<(NodeIndex, FirEdge)> = vec![];
        for eid in fame5.graph.edges_directed(reg_id, Outgoing) {
            let mut edge = fame5.graph.edge_weight(eid.id()).unwrap().clone();
            edge.src = Expr::Reference(Reference::RefDot(
                Box::new(rport_ref.clone()), Identifier::Name("data".to_string())));
            let ep = fame5.graph.edge_endpoints(eid.id()).unwrap();
            sink_edges.push((ep.1, edge));
        }

        for (dst, edge) in sink_edges {
            fame5.graph.add_edge(reg_threaded_id, dst, edge);
        }

        // wen
        //         |   incoming edge          | x incoming edge      |
        // reginit |   mux(hostreset, 1, 1)   | mux(hostreset, 1, 0) |
        // reg     |   1                      | 0                    |
        let drivers: Vec<EdgeIndex> = fame5.graph.edges_directed(reg_id, Incoming).filter(|eid| {
            let edge = fame5.graph.edge_weight(eid.id()).unwrap();
            edge.et == FirEdgeType::Wire
        })
        .map(|eid| eid.id())
        .collect();

        let non_hostreset_wen = if drivers.len() > 0 {
            1
        } else {
            0
        };
        let (wen_id, wen_expr) = fame5.add_uint_literal(non_hostreset_wen, 1);

        if node.nt == FirNodeType::RegReset {
            let hostreset_edge = find_edge_with_type(&fame5, reg_id, FirEdgeType::Reset, Incoming).unwrap();
            let hostreset_id = fame5.graph.edge_endpoints(hostreset_edge).unwrap().0;
            let hostreset_edge = fame5.graph.edge_weight(hostreset_edge).unwrap().clone();
            let hostreset_expr = hostreset_edge.src.clone();

            // Create mux node for RegInit write enable
            let (one_const_id, one_const_expr) = fame5.add_uint_literal(1, 1);
            let (mux_id, mux_expr) = fame5.add_mux(
                hostreset_id,
                hostreset_expr.clone(),
                one_const_id,
                one_const_expr,
                wen_id,
                wen_expr,
            );

            // Connect mux output to write port enable
            fame5.add_wire(
                mux_id,
                mux_expr,
                reg_threaded_id,
                Some(Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("en".to_string()))),
            );


            for eid in drivers.iter() {
                // Find initial value for RegReset
                let init_edge = find_edge_with_type(&fame5, reg_id, FirEdgeType::InitValue, Incoming).unwrap();
                let init_id = fame5.graph.edge_endpoints(init_edge).unwrap().0;
                let init_expr = fame5.graph.edge_weight(init_edge).unwrap().clone().src;

                // Connect drivers to data mux false input (when hostreset is false)
                let driver_id = fame5.graph.edge_endpoints(*eid).unwrap().0;
                let driver_expr = fame5.graph.edge_weight(*eid).unwrap().clone().src;

                // Create mux node for data selection
                let (data_mux_id, data_mux_expr) = fame5.add_mux(
                    hostreset_id,
                    hostreset_expr.clone(),
                    init_id,
                    init_expr,
                    driver_id,
                    driver_expr);

                // Connect data mux output to write port data
                fame5.add_wire(
                    data_mux_id,
                    data_mux_expr,
                    reg_threaded_id,
                    Some(Reference::RefDot(
                        Box::new(wport_ref.clone()),
                        Identifier::Name("data".to_string()))));
           }
        } else {
            fame5.add_wire(
                wen_id,
                wen_expr,
                reg_threaded_id,
                Some(Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("en".to_string()))),
            );

            for eid in drivers.iter() {
                let driver_id = fame5.graph.edge_endpoints(*eid).unwrap().0;
                let mut driver_edge = fame5.graph.edge_weight(*eid).unwrap().clone();
                driver_edge.dst = Some(Reference::RefDot(
                        Box::new(wport_ref.clone()),
                        Identifier::Name("data".to_string())));
                fame5.graph.add_edge(driver_id, reg_threaded_id, driver_edge);
            }
        }

        // Remove the node now
        fame5.graph.remove_node(reg_id);
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

struct MemPortInfo<'a> {
    orig_id: NodeIndex,
    new_id: NodeIndex,
    mem: &'a Identifier,
    port: &'a Identifier
}

impl <'a> MemPortInfo<'a> {
    fn field_ref(&self, field: &Identifier) -> Reference {
        let port_ref = Reference::RefDot(Box::new(
                Reference::RefDot(
                    Box::new(Reference::Ref(self.mem.clone())),
                    self.port.clone())),
                    field.clone());
        port_ref
    }
}

fn find_memport_edge<'a>(
    fg: &FirGraph,
    memport: &'a MemPortInfo,
    field: &Identifier,
    dir: Direction
) -> Option<EdgeIndex> {
    let field_ref = memport.field_ref(field);
    for eid in fg.graph.edges_directed(memport.orig_id, dir) {
        let edge = fg.graph.edge_weight(eid.id()).unwrap();
        if let Some(dst_ref) = edge.dst.as_ref() {
            if dst_ref == &field_ref {
                return Some(eid.id());
            }
        }
    }
    None
}

fn add_edge_input_mem_edge<'a>(
    fg: &mut FirGraph,
    field: Identifier,
    memport: &'a MemPortInfo
)
{
    if let Some(eid) = find_memport_edge(fg, memport, &field, Incoming) {
        let ep = fg.graph.edge_endpoints(eid).unwrap();
        let mut edge = fg.graph.edge_weight(eid).unwrap().clone();
        edge.dst = Some(memport.field_ref(&field));
        fg.graph.add_edge(ep.0, memport.new_id, edge);
    }
}

fn wport_connections<'a>(
    fg: &mut FirGraph,
    memport: &'a MemPortInfo,
    thread_idx_id: NodeIndex,
)
{
    let addr_driver = find_memport_edge(fg, memport, &Identifier::Name("addr".to_string()), Incoming);

    add_edge_input_mem_edge(fg, Identifier::Name("en".to_string()), memport);
    add_edge_input_mem_edge(fg, Identifier::Name("mask".to_string()), memport);
    add_edge_input_mem_edge(fg, Identifier::Name("data".to_string()), memport);

}

pub fn pipeline_memories(
    fg: &FirGraph,
    nthreads: u32,
    thread_idx_name: &Identifier,
    thread_idx_id: NodeIndex,
) -> FirGraph {
    let mut pipelined = fg.clone();

    // Find all memory nodes with read latency > 0
    let mut mem_ids: Vec<NodeIndex> = pipelined.graph.node_indices().filter(|id| {
        let node = pipelined.graph.node_weight(*id).unwrap();
        if let FirNodeType::Memory(_, rlat, wlat, _, _) = node.nt {
            assert!(rlat <= 1);
            assert!(wlat == 0);
            rlat > 0
        } else {
            false
        }
    }).collect();

    mem_ids.sort();

    for &mem_id in mem_ids.iter().rev() {
        let node = pipelined.graph.node_weight(mem_id).unwrap().clone();
        if let FirNodeType::Memory(depth, rlat, wlat, ports, ruw) = &node.nt {
            let mem_name = node.name.as_ref().unwrap().clone();

            // Create new memory with capacity multiplied by nthreads
            let new_depth = depth * nthreads;
            let new_mem_type = memory_type_from_ports(ports, new_depth, &node.ttree.as_ref().unwrap().to_type());
            let new_mem = FirNode::new(
                Some(mem_name.clone()),
                FirNodeType::Memory(new_depth, *rlat, *wlat, ports.clone(), ruw.clone()),
                Some(TypeTree::build_from_type(&new_mem_type, TypeDirection::Outgoing))
            );

            let new_mem_id = pipelined.graph.add_node(new_mem);

            for port in ports {
                match port.as_ref() {
                    MemoryPort::Read(name) => {
                    },
                    MemoryPort::Write(p) => {
// let addr_driver = find_memport_edge(fg, &mem_id, &mem_name, p, &Identifier::Name("addr".to_string()), Incoming);
// if let Some(en_eid) = find_memport_edge(fg, &mem_id, &mem_name, p, &Identifier::Name("en".to_string()), Incoming) {
// }


// let data_driver = ;
// let mask_driver = ;
                    },
                    MemoryPort::ReadWrite(name) => {
                    },
                }
            }


            // Remove the original memory node
            pipelined.graph.remove_node(mem_id);
        }
    }

    pipelined
}
