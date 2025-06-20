use crate::passes::fir::fame::fame5::*;
use crate::passes::fir::from_ast::memory_type_from_ports;
use crate::ir::typetree::typetree::TypeTree;
use crate::ir::typetree::tnode::TypeDirection;
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

pub fn multithread_module(
    fg: &FirGraph,
    nthreads: u32,
    host_clock: &Identifier,
    host_reset: &Identifier
) -> FirGraph {
    let mut fame5 = fg.clone();

    let thread_idx_id = add_thread_idx_reg(&mut fame5, nthreads, host_clock, host_reset);
    let thread_idx_name = fame5.graph.node_weight(thread_idx_id).unwrap().name.as_ref().unwrap().clone();
    add_thread_idx_update(&mut fame5, thread_idx_id, nthreads, log2_ceil(nthreads));

    let mut reg_ids: Vec<NodeIndex> = fg.graph.node_indices().filter(|id| {
        let node = fg.graph.node_weight(*id).unwrap();
        node.nt == FirNodeType::Reg ||
            node.nt == FirNodeType::RegReset
    }).collect();

    for &reg_id in reg_ids.iter() {
        let rport_name = Identifier::Name("rd".to_string());
        let rport = MemoryPort::Read(rport_name.clone());

        let wport_name = Identifier::Name("wr".to_string());
        let wport = MemoryPort::Write(wport_name.clone());
        let ports = vec![
            Box::new(rport),
            Box::new(wport),
        ];


        let node = fg.graph.node_weight(reg_id).unwrap();
        let reg_name = node.name.as_ref().unwrap();
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

        let clock_eid = find_edge_with_type(fg, reg_id, FirEdgeType::Clock, Incoming).unwrap();
        let clock_id = fg.graph.edge_endpoints(clock_eid).unwrap().0;
        let clock_edge = fg.graph.edge_weight(clock_eid).unwrap().clone();

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
        for eid in fg.graph.edges_directed(reg_id, Outgoing) {
            let mut edge = fg.graph.edge_weight(eid.id()).unwrap().clone();
            edge.src = Expr::Reference(Reference::RefDot(
                Box::new(rport_ref.clone()), Identifier::Name("data".to_string())));

            let ep = fg.graph.edge_endpoints(eid.id()).unwrap();
            fame5.graph.add_edge(reg_threaded_id, ep.1, edge);
        }

        // wen
        //         |   incoming edge          | x incoming edge      |
        // reginit |   mux(hostreset, 1, 1)   | mux(hostreset, 1, 0) |
        // reg     |   1                      | 0                    |
        let drivers: Vec<EdgeIndex> = fg.graph.edges_directed(reg_id, Incoming).filter(|eid| {
            let edge = fg.graph.edge_weight(eid.id()).unwrap();
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
            let hostreset_edge = find_edge_with_type(fg, reg_id, FirEdgeType::Reset, Incoming).unwrap();
            let hostreset_id = fg.graph.edge_endpoints(hostreset_edge).unwrap().0;
            let hostreset_edge = fg.graph.edge_weight(hostreset_edge).unwrap().clone();
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
                let init_edge = find_edge_with_type(fg, reg_id, FirEdgeType::InitValue, Incoming).unwrap();
                let init_id = fg.graph.edge_endpoints(init_edge).unwrap().0;
                let init_expr = fg.graph.edge_weight(init_edge).unwrap().clone().src;

                // Connect drivers to data mux false input (when hostreset is false)
                let driver_id = fg.graph.edge_endpoints(*eid).unwrap().0;
                let driver_expr = fg.graph.edge_weight(*eid).unwrap().clone().src;

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
                let driver_id = fg.graph.edge_endpoints(*eid).unwrap().0;
                let mut driver_edge = fg.graph.edge_weight(*eid).unwrap().clone();
                driver_edge.dst = Some(Reference::RefDot(
                        Box::new(wport_ref.clone()),
                        Identifier::Name("data".to_string())));
                fame5.graph.add_edge(driver_id, reg_threaded_id, driver_edge);
            }
        }
    }

    reg_ids.sort();
    for &reg_id in reg_ids.iter().rev() {
        fame5.graph.remove_node(reg_id);
    }

    fame5
}
