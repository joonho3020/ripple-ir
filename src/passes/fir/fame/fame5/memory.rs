use crate::passes::fir::fame::fame5::*;
use crate::passes::fir::from_ast::memory_type_from_ports;
use crate::ir::typetree::typetree::TypeTree;
use crate::ir::typetree::tnode::TypeDirection;
use petgraph::graph::NodeIndex;
use rusty_firrtl::{Identifier, MemoryPort, PrimOp2Expr, ChirrtlMemoryReadUnderWrite};
use petgraph::graph::EdgeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction::{Incoming, Outgoing};

pub fn duplicate_memory(
    fame5: &mut FirGraph,
    nthreads: u32,
    thread_idx_id: NodeIndex,
    mem_id: NodeIndex
)
{
    let node = fame5.graph.node_weight(mem_id).unwrap().clone();
    if let FirNodeType::Memory(depth, rlat, wlat, ports, ruw) = &node.nt {
        let mem_name = node.name.as_ref().unwrap().clone();
        let mem_ttree = node.ttree.as_ref().unwrap().clone();

        // Create new memory with capacity multiplied by nthreads
        let new_depth = depth * nthreads;
        let new_mem = FirNode::new(
            Some(mem_name.clone()),
            FirNodeType::Memory(new_depth, *rlat, *wlat, ports.clone(), ruw.clone()),
            Some(mem_ttree)
        );

        let new_mem_id = fame5.graph.add_node(new_mem);

        for port in ports {
            match port.as_ref() {
                MemoryPort::Read(port_name) => {
                    let mport = MemPortInfo::new(mem_id, new_mem_id, thread_idx_id, &mem_name, port_name, nthreads, *depth);
                    rport_connections(fame5, &mport);
                },
                MemoryPort::Write(port_name) => {
                    let mport = MemPortInfo::new(mem_id, new_mem_id, thread_idx_id, &mem_name, port_name, nthreads, *depth);
                    wport_connections(fame5, &mport);
                },
                MemoryPort::ReadWrite(port_name) => {
                    let mport = MemPortInfo::new(mem_id, new_mem_id, thread_idx_id, &mem_name, port_name, nthreads, *depth);
                    rwport_connections(fame5, &mport);
                },
            }
        }
    }
}

struct MemPortInfo<'a> {
    /// Original memory node index
    orig_id: NodeIndex,

    /// New memory node index
    new_id: NodeIndex,

    /// Thread index node index
    thread_idx_id: NodeIndex,

    /// Name of the memory
    mem: &'a Identifier,

    /// Name of the port that we are currently handling
    port: &'a Identifier,

    /// Number of threads
    nthreads: u32,

    /// Memory depth
    depth: u32,
}

impl <'a> MemPortInfo<'a> {
    fn new(
        orig_id: NodeIndex,
        new_id: NodeIndex,
        thread_idx_id: NodeIndex,
        mem: &'a Identifier,
        port: &'a Identifier,
        nthreads: u32,
        depth: u32
    ) -> Self {
        Self { orig_id, new_id, thread_idx_id, mem, port, nthreads, depth }
    }

    fn field_ref(&self, field: &Identifier) -> Reference {
        let port_ref = Reference::RefDot(Box::new(
                Reference::RefDot(
                    Box::new(Reference::Ref(self.mem.clone())),
                    self.port.clone())),
                    field.clone());
        port_ref
    }
}

fn find_memport_incoming_edge<'a>(
    fg: &FirGraph,
    memport: &'a MemPortInfo,
    field: &Identifier
) -> Option<EdgeIndex> {
    let field_ref = memport.field_ref(field);
    for eid in fg.graph.edges_directed(memport.orig_id, Incoming) {
        let edge = fg.graph.edge_weight(eid.id()).unwrap();
        if let Some(dst_ref) = edge.dst.as_ref() {
            if dst_ref == &field_ref {
                return Some(eid.id());
            }
        }
    }
    None
}

fn find_memport_outgoing_edges<'a>(
    fg: &FirGraph,
    memport: &'a MemPortInfo,
    field: &Identifier
) -> Vec<EdgeIndex> {
    let mut ret = vec![];
    for eid in fg.graph.edges_directed(memport.orig_id, Outgoing) {
        let field_ref = memport.field_ref(field);
        let edge = fg.graph.edge_weight(eid.id()).unwrap();
        if edge.src == Expr::Reference(field_ref) {
            ret.push(eid.id());
        }
    }
    ret
}

fn add_input_mem_edge<'a>(
    fg: &mut FirGraph,
    field: Identifier,
    memport: &'a MemPortInfo
)
{
    if let Some(eid) = find_memport_incoming_edge(fg, memport, &field) {
        let ep = fg.graph.edge_endpoints(eid).unwrap();
        let mut edge = fg.graph.edge_weight(eid).unwrap().clone();
        edge.dst = Some(memport.field_ref(&field));
        fg.graph.add_edge(ep.0, memport.new_id, edge);
    }
}

/// Check if a number is a power of two
fn is_power_of_two(n: u32) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn add_input_addr_edge<'a>(
    fg: &mut FirGraph,
    addr: Identifier,
    memport: &'a MemPortInfo,
)
{
    let thread_idx_id = memport.thread_idx_id;
    let nthreads = memport.nthreads;
    let original_depth = memport.depth;

    let addr_driver = find_memport_incoming_edge(fg, memport, &addr);
    if let Some(addr_eid) = addr_driver {
        let ep = fg.graph.edge_endpoints(addr_eid).unwrap();
        let addr_driver_id = ep.0;
        let addr_driver_expr = fg.graph.edge_weight(addr_eid).unwrap().src.clone();

        // Get thread_idx name from the node
        let thread_idx_name = fg.graph.node_weight(thread_idx_id).unwrap().name.as_ref().unwrap().clone();
        let thread_idx_expr = Expr::Reference(Reference::Ref(thread_idx_name));

        if is_power_of_two(original_depth) {
            // If memory depth is a power of two, use concatenation (Cat operator)
            // This is more efficient as it just concatenates thread_idx with the original address
            let (cat_id, cat_expr) = fg.add_primop2(
                PrimOp2Expr::Cat,
                thread_idx_id,
                thread_idx_expr,
                addr_driver_id,
                addr_driver_expr);

            // Connect concatenated address to the new memory port
            fg.add_wire(
                cat_id,
                cat_expr,
                memport.new_id,
                Some(memport.field_ref(&addr)));
        } else {
            // If memory depth is not a power of two, use: address + thread_idx * original_depth
            // First, multiply thread_idx by original_depth
            let (depth_const_id, depth_expr) = fg.add_uint_literal(original_depth, log2_ceil(original_depth * nthreads));
            let (mul_id, mul_expr) = fg.add_primop2(
                PrimOp2Expr::Mul,
                thread_idx_id,
                thread_idx_expr,
                depth_const_id,
                depth_expr);

            // Then add the original address to the result
            let (add_id, add_expr) = fg.add_primop2(
                PrimOp2Expr::Add,
                mul_id,
                mul_expr,
                addr_driver_id,
                addr_driver_expr);

            // Connect the computed address to the new memory port
            fg.add_wire(
                add_id,
                add_expr,
                memport.new_id,
                Some(memport.field_ref(&addr)));
        }
    }
}

/// - rdata: Name of the read data port
fn add_output_rdata_edge<'a>(
    fg: &mut FirGraph,
    rdata: Identifier,
    memport: &'a MemPortInfo
)
{
    let nthreads = memport.nthreads;
    let thread_idx_id = memport.thread_idx_id;

    // Get the original memory node to check read latency
    let orig_mem_node = fg.graph.node_weight(memport.orig_id).unwrap();
    let read_latency = if let FirNodeType::Memory(_, rlat, ..) = &orig_mem_node.nt {
        rlat
    } else {
        panic!("Expected memory node");
    };

    if *read_latency == 0 {
        // If read latency is zero, just connect the rdata port to the sinks
        for eid in find_memport_outgoing_edges(fg, memport, &rdata) {
            let ep = fg.graph.edge_endpoints(eid).unwrap();
            let mut edge = fg.graph.edge_weight(eid).unwrap().clone();
            edge.src = Expr::Reference(memport.field_ref(&rdata));
            fg.graph.add_edge(memport.new_id, ep.1, edge);
        }
    } else {
        // Create combinational memory with depth nthreads
        let rport_name = Identifier::Name("rd".to_string());
        let wport_name = Identifier::Name("wr".to_string());
        let ports = vec![
            Box::new(MemoryPort::Read(rport_name.clone())),
            Box::new(MemoryPort::Write(wport_name.clone())),
        ];

        // Get the data type from the original memory
        let mem_type = fg.memory_base_type(memport.orig_id);
        let comb_mem_type = memory_type_from_ports(&ports, nthreads, &mem_type);
        let comb_mem_name = Identifier::Name(format!("{}_{}_fame5_pipeline", memport.mem, memport.port));
        let comb_mem = FirNode::new(
            Some(comb_mem_name.clone()),
            FirNodeType::Memory(nthreads, 0, 1, ports, ChirrtlMemoryReadUnderWrite::Undefined),
            Some(TypeTree::build_from_type(&comb_mem_type, TypeDirection::Outgoing)));

        let comb_mem_id = fg.graph.add_node(comb_mem);

        // Create references for the combinational memory ports
        let rport_ref = Reference::RefDot(
            Box::new(Reference::Ref(comb_mem_name.clone())),
            rport_name.clone());

        let wport_ref = Reference::RefDot(
            Box::new(Reference::Ref(comb_mem_name.clone())),
            wport_name.clone());

        // Connect read port enable to 1
        let (one_const_id, one_expr) = fg.add_uint_literal(1, 1);
        fg.add_wire(
            one_const_id,
            one_expr,
            comb_mem_id,
            Some(Reference::RefDot(Box::new(rport_ref.clone()), Identifier::Name("en".to_string()))),
        );

        // Connect read port clock (same as original read port clock)
        if let Some(clk_eid) = find_memport_incoming_edge(fg, memport, &Identifier::Name("clk".to_string())) {
            let ep = fg.graph.edge_endpoints(clk_eid).unwrap();
            let clk_edge = fg.graph.edge_weight(clk_eid).unwrap().clone();
            let mut new_clk_edge = clk_edge.clone();
            new_clk_edge.dst = Some(Reference::RefDot(Box::new(rport_ref.clone()), Identifier::Name("clk".to_string())));
            fg.graph.add_edge(ep.0, comb_mem_id, new_clk_edge);
        }

        // Connect read port address to thread_idx
        let thread_idx_name = fg.graph.node_weight(thread_idx_id).unwrap().name.as_ref().unwrap().clone();
        let thread_idx_expr = Expr::Reference(Reference::Ref(thread_idx_name.clone()));
        fg.add_wire(
            thread_idx_id,
            thread_idx_expr,
            comb_mem_id,
            Some(Reference::RefDot(Box::new(rport_ref.clone()), Identifier::Name("addr".to_string()))),
        );

        // Connect write port enable (same as original read port enable)
        if let Some(en_eid) = find_memport_incoming_edge(fg, memport, &Identifier::Name("en".to_string())) {
            let ep = fg.graph.edge_endpoints(en_eid).unwrap();
            let en_edge = fg.graph.edge_weight(en_eid).unwrap().clone();
            let mut new_en_edge = en_edge.clone();
            new_en_edge.dst = Some(Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("en".to_string())));
            fg.graph.add_edge(ep.0, comb_mem_id, new_en_edge);
        }

        // Connect write port mask to 1
        let (mask_one_id, mask_one_expr) = fg.add_uint_literal(1, 1);
        fg.add_wire(
            mask_one_id,
            mask_one_expr,
            comb_mem_id,
            Some(Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("mask".to_string()))));

        // Connect write port clock (same as original read port clock)
        if let Some(clk_eid) = find_memport_incoming_edge(fg, memport, &Identifier::Name("clk".to_string())) {
            let ep = fg.graph.edge_endpoints(clk_eid).unwrap();
            let clk_edge = fg.graph.edge_weight(clk_eid).unwrap().clone();
            let mut new_clk_edge = clk_edge.clone();
            new_clk_edge.dst = Some(Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("clk".to_string())));
            fg.graph.add_edge(ep.0, comb_mem_id, new_clk_edge);
        }

        // Create a register to store the previous cycle's thread_idx value
        let thread_idx_bits = log2_ceil(nthreads);
        let prev_thread_idx_name = Identifier::Name(format!("{}_{}_prev_thread_idx", memport.mem, memport.port));
        let prev_thread_idx_reg = FirNode::new(
            Some(prev_thread_idx_name.clone()),
            FirNodeType::Reg,
            Some(TypeTree::build_from_ground_type(GroundType::UInt(Some(Width(thread_idx_bits)))))
        );
        let prev_thread_idx_id = fg.graph.add_node(prev_thread_idx_reg);

        // Connect the register's clock (same as write port clock)
        if let Some(clk_eid) = find_memport_incoming_edge(fg, memport, &Identifier::Name("clk".to_string())) {
            let ep = fg.graph.edge_endpoints(clk_eid).unwrap();
            let clk_edge = fg.graph.edge_weight(clk_eid).unwrap().clone();

            // Connect clock to prev_thread_idx register
            let mut new_clk_edge = clk_edge.clone();
            new_clk_edge.et = FirEdgeType::Clock;
            new_clk_edge.dst = None; // Clock edges don't have destination references
            fg.graph.add_edge(ep.0, prev_thread_idx_id, new_clk_edge);
        }

        // Connect thread_idx to the register's input (next value)
        let thread_idx_name = fg.graph.node_weight(thread_idx_id).unwrap().name.as_ref().unwrap().clone();
        let thread_idx_expr = Expr::Reference(Reference::Ref(thread_idx_name.clone()));

        // Create logic to determine when to update prev_thread_idx
        // We want to update when prev_thread_idx == thread_idx - 1 (with wraparound)

        // First, compute thread_idx - 1
        let (one_const_id, one_expr) = fg.add_uint_literal(1, thread_idx_bits);
        let (sub_id, sub_expr) = fg.add_primop2(
            PrimOp2Expr::Sub,
            thread_idx_id,
            thread_idx_expr.clone(),
            one_const_id,
            one_expr,
        );

        // Create constant for nthreads - 1 (for wraparound case)
        let (nthreads_minus_one_id, nthreads_minus_one_expr) = fg.add_uint_literal(nthreads - 1, thread_idx_bits);

        // Check if thread_idx == 0
        let (zero_const_id, zero_expr) = fg.add_uint_literal(0, thread_idx_bits);
        let (eq_zero_id, eq_zero_expr) = fg.add_primop2(
            PrimOp2Expr::Eq,
            thread_idx_id,
            thread_idx_expr.clone(),
            zero_const_id,
            zero_expr,
        );

        // Select between thread_idx - 1 and nthreads - 1 based on whether thread_idx == 0
        let (expected_prev_id, expected_prev_expr) = fg.add_mux(
            eq_zero_id,
            eq_zero_expr,
            nthreads_minus_one_id,
            nthreads_minus_one_expr,
            sub_id,
            sub_expr,
        );

        // Compare prev_thread_idx with the expected value
        let prev_thread_idx_expr = Expr::Reference(Reference::Ref(prev_thread_idx_name.clone()));
        let (eq_expected_id, eq_expected_expr) = fg.add_primop2(
            PrimOp2Expr::Eq,
            prev_thread_idx_id,
            prev_thread_idx_expr.clone(),
            expected_prev_id,
            expected_prev_expr,
        );

        // Mux to select between thread_idx (when equal) and prev_thread_idx (when not equal)
        let (mux_id, mux_expr) = fg.add_mux(
            eq_expected_id,
            eq_expected_expr.clone(),
            thread_idx_id,
            thread_idx_expr,
            prev_thread_idx_id,
            prev_thread_idx_expr,
        );

        // Connect the mux output to the register input
        let reg_input_edge = FirEdge::new(mux_expr, Some(Reference::Ref(prev_thread_idx_name.clone())), FirEdgeType::Wire);
        fg.graph.add_edge(mux_id, prev_thread_idx_id, reg_input_edge);

        // Update the skip_update register: set to false if we updated prev_thread_idx, true otherwise
        let (false_const_id, false_expr) = fg.add_uint_literal(0, 1);
        let (true_const_id, true_expr) = fg.add_uint_literal(1, 1);
        let (skip_mux_id, skip_mux_expr) = fg.add_mux(
            eq_expected_id,
            eq_expected_expr.clone(),
            false_const_id,
            false_expr,
            true_const_id,
            true_expr,
        );

        // Connect the register's output (previous cycle's thread_idx) to write port address
        let prev_thread_idx_expr = Expr::Reference(Reference::Ref(prev_thread_idx_name.clone()));

        // Compute prev_thread_idx + 1
        let (one_const_id, one_expr) = fg.add_uint_literal(1, thread_idx_bits);
        let (add_id, add_expr) = fg.add_primop2(
            PrimOp2Expr::Add,
            prev_thread_idx_id,
            prev_thread_idx_expr.clone(),
            one_const_id,
            one_expr);

        // Check if prev_thread_idx + 1 equals nthreads (for wraparound)
        let (nthreads_const_id, nthreads_expr) = fg.add_uint_literal(nthreads - 1, thread_idx_bits);
        let (eq_nthreads_id, eq_nthreads_expr) = fg.add_primop2(
            PrimOp2Expr::Eq,
            prev_thread_idx_id,
            prev_thread_idx_expr.clone(),
            nthreads_const_id,
            nthreads_expr);

        // Select between prev_thread_idx + 1 and 0 based on wraparound
        let (zero_const_id, zero_expr) = fg.add_uint_literal(0, thread_idx_bits);
        let (wrapped_id, wrapped_expr) = fg.add_mux(
            eq_nthreads_id,
            eq_nthreads_expr,
            zero_const_id,
            zero_expr,
            add_id,
            add_expr);

        // Final mux to select between wrapped value (when skip_update is true) and prev_thread_idx (when false)
        let (final_mux_id, final_mux_expr) = fg.add_mux(
            skip_mux_id,
            skip_mux_expr,
            wrapped_id,
            wrapped_expr,
            prev_thread_idx_id,
            prev_thread_idx_expr);

        // Connect the final mux output to write port address
        fg.add_wire(
            final_mux_id,
            final_mux_expr,
            comb_mem_id,
            Some(Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("addr".to_string()))),
        );

        // Connect write port data from original memory rdata
        let write_src = Expr::Reference(memport.field_ref(&rdata));
        let write_dst = Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("data".to_string()));
        let write_edge = FirEdge::new(write_src, Some(write_dst), FirEdgeType::Wire);
        fg.graph.add_edge(memport.new_id, comb_mem_id, write_edge);

        // Connect combinational memory read data to original sinks
        for eid in find_memport_outgoing_edges(fg, memport, &rdata) {
            let ep = fg.graph.edge_endpoints(eid).unwrap();
            let mut edge = fg.graph.edge_weight(eid).unwrap().clone();
            edge.src = Expr::Reference(
                Reference::RefDot(
                    Box::new(rport_ref.clone()),
                    Identifier::Name("data".to_string())));
            fg.graph.add_edge(comb_mem_id, ep.1, edge);
        }
    }
}

fn wport_connections<'a>(fg: &mut FirGraph, memport: &'a MemPortInfo) {
    add_input_addr_edge(fg, Identifier::Name("addr".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("en".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("clk".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("data".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("mask".to_string()), memport);
}

fn rwport_connections<'a>(fg: &mut FirGraph, memport: &'a MemPortInfo) {
    add_input_addr_edge(fg, Identifier::Name("addr".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("en".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("clk".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("wmask".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("wdata".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("wmode".to_string()), memport);
    add_output_rdata_edge(fg, Identifier::Name("rdata".to_string()), memport);
}

fn rport_connections<'a>(fg: &mut FirGraph, memport: &'a MemPortInfo) {
    add_input_addr_edge(fg, Identifier::Name("addr".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("en".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("clk".to_string()), memport);
    add_output_rdata_edge(fg, Identifier::Name("data".to_string()), memport);
}
