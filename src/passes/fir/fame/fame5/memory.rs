use crate::passes::fir::fame::fame5::*;
use crate::passes::fir::from_ast::memory_type_from_ports;
use crate::ir::typetree::typetree::TypeTree;
use crate::ir::typetree::tnode::{TypeDirection, GroundType};
use petgraph::graph::NodeIndex;
use rusty_firrtl::{Identifier, MemoryPort, PrimOp1Expr, PrimOp2Expr, ChirrtlMemoryReadUnderWrite, Width};
use petgraph::graph::EdgeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction::{Incoming, Outgoing};

pub fn duplicate_memory(
    fame5: &mut FirGraph,
    nthreads: u32,
    thread_idx_id: NodeIndex,
    mem_id: NodeIndex,
    host_clock_id: NodeIndex,
    host_clock: &Identifier
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
                    let mport = MemPortInfo::new(mem_id, new_mem_id, thread_idx_id, &mem_name, port_name, nthreads, *depth, *rlat, host_clock_id, host_clock);
                    rport_connections(fame5, &mport);
                },
                MemoryPort::Write(port_name) => {
                    let mport = MemPortInfo::new(mem_id, new_mem_id, thread_idx_id, &mem_name, port_name, nthreads, *depth, *rlat, host_clock_id, host_clock);
                    wport_connections(fame5, &mport);
                },
                MemoryPort::ReadWrite(port_name) => {
                    let mport = MemPortInfo::new(mem_id, new_mem_id, thread_idx_id, &mem_name, port_name, nthreads, *depth, *rlat, host_clock_id, host_clock);
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

    /// Read latency
    read_latency: u32,

    /// Host clock node index
    host_clock_id: NodeIndex,

    /// Host clock name
    host_clock: &'a Identifier,
}

impl <'a> MemPortInfo<'a> {
    fn new(
        orig_id: NodeIndex,
        new_id: NodeIndex,
        thread_idx_id: NodeIndex,
        mem: &'a Identifier,
        port: &'a Identifier,
        nthreads: u32,
        depth: u32,
        read_latency: u32,
        host_clock_id: NodeIndex,
        host_clock: &'a Identifier
    ) -> Self {
        Self { orig_id, new_id, thread_idx_id, mem, port, nthreads, depth, read_latency, host_clock_id, host_clock }
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

/// Find output node for a read data channel
/// The output channel name pattern is typically "io_reads_X_data_source" where X is the port index
/// We try to match based on the port name or by finding output nodes with "reads" and "data_source"
fn find_output_node_for_read_data(fg: &FirGraph, memport: &MemPortInfo) -> Option<NodeIndex> {
    let port_name_str = if let Identifier::Name(s) = memport.port { s.clone() } else { return None };

    // Try to find output node that matches the port name pattern
    // First, try exact match with port name + "_data_source"
    let exact_pattern = format!("{}_data_source", port_name_str);
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
        if let FirNodeType::Output = node.nt {
            if let Some(name) = &node.name {
                if let Identifier::Name(name_str) = name {
                    if name_str == &exact_pattern {
                        return Some(id);
                    }
                }
            }
        }
    }

    // If exact match fails, try to find output nodes with "reads" and "data_source" in the name
    // and match by port index if we can extract it from the port name
    let mut candidates = Vec::new();
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
        if let FirNodeType::Output = node.nt {
            if let Some(name) = &node.name {
                if let Identifier::Name(name_str) = name {
                    if name_str.contains("reads") && name_str.contains("data_source") {
                        candidates.push((id, name_str.clone()));
                    }
                }
            }
        }
    }

    // If we have candidates, try to match by port index
    // Extract number from port name (e.g., "reads_0" -> 0)
    if let Some(port_idx) = extract_port_index(&port_name_str) {
        for (id, name_str) in &candidates {
            if let Some(output_idx) = extract_port_index(name_str) {
                if port_idx == output_idx {
                    return Some(*id);
                }
            }
        }
    }

    // If no match by index, return first candidate if only one exists
    if candidates.len() == 1 {
        return Some(candidates[0].0);
    }

    None
}

/// Extract port index from a name (e.g., "reads_0" -> Some(0), "io_reads_1_data_source" -> Some(1))
fn extract_port_index(name: &str) -> Option<u32> {
    // Try to find pattern like "reads_0" or "reads_1" in the name
    if let Some(reads_pos) = name.find("reads_") {
        let after_reads = &name[reads_pos + 6..]; // Skip "reads_"
        // Find the next underscore or end of string
        let end_pos = after_reads.find('_').unwrap_or(after_reads.len());
        let num_str = &after_reads[..end_pos];
        if let Ok(num) = num_str.parse::<u32>() {
            return Some(num);
        }
    }
    // Fallback: try to find any number after an underscore
    if let Some(underscore_pos) = name.rfind('_') {
        let after_underscore = &name[underscore_pos + 1..];
        // Check if it's a number (for cases like "port_0")
        if let Ok(num) = after_underscore.parse::<u32>() {
            return Some(num);
        }
    }
    None
}

/// Create a register with host clock that captures the given expression on each clock edge.
/// Returns the register's node index and reference expression.
fn host_reg_next(
    fg: &mut FirGraph,
    name: Identifier,
    input_id: NodeIndex,
    input_expr: Expr,
    host_clock_id: NodeIndex,
    host_clock: &Identifier,
    width: u32
) -> (NodeIndex, Expr) {
    // Create register node
    let reg_ttree = TypeTree::build_from_ground_type(GroundType::UInt(Some(Width(width))));
    let reg_node = FirNode::new(
        Some(name.clone()),
        FirNodeType::Reg,
        Some(reg_ttree)
    );
    let reg_id = fg.graph.add_node(reg_node);

    // Connect host clock to register
    let clock_edge = FirEdge::new(
        Expr::Reference(Reference::Ref(host_clock.clone())),
        None,
        FirEdgeType::Clock
    );
    fg.graph.add_edge(host_clock_id, reg_id, clock_edge);

    // Connect input to register
    let input_edge = FirEdge::new(
        input_expr,
        Some(Reference::Ref(name.clone())),
        FirEdgeType::Wire
    );
    fg.graph.add_edge(input_id, reg_id, input_edge);

    let reg_expr = Expr::Reference(Reference::Ref(name));
    (reg_id, reg_expr)
}

/// Create a pipeline of registers
/// Returns the final register's node index and reference expression
fn pipeline(
    fg: &mut FirGraph,
    depth: u32,
    input_id: NodeIndex,
    input_expr: Expr,
    base_name: &str,
    host_clock_id: NodeIndex,
    host_clock: &Identifier,
    width: u32
) -> (NodeIndex, Expr) {
    if depth == 0 {
        return (input_id, input_expr);
    }

    let mut current_id = input_id;
    let mut current_expr = input_expr;

    for idx in 1..=depth {
        let reg_name = fg.namespace.new_name(&format!("{}_p{}", base_name, idx));
        let (new_id, new_expr) = host_reg_next(
            fg,
            reg_name,
            current_id,
            current_expr,
            host_clock_id,
            host_clock,
            width
        );
        current_id = new_id;
        current_expr = new_expr;
    }

    (current_id, current_expr)
}

/// Create a buffer memory for storing read data in the read latency > 0 case.
/// Returns the buffer memory node index.
fn create_buffer_memory(
    fg: &mut FirGraph,
    name: Identifier,
    data_type: &TypeTree,
    nthreads: u32,
    buffer_read_latency: u32
) -> NodeIndex {
    let rport_name = Identifier::Name("r".to_string());
    let wport_name = Identifier::Name("w".to_string());
    let ports = vec![
        Box::new(MemoryPort::Read(rport_name)),
        Box::new(MemoryPort::Write(wport_name)),
    ];

    let mem_type = memory_type_from_ports(&ports, nthreads, &data_type.to_type());
    let mem_node = FirNode::new(
        Some(name),
        FirNodeType::Memory(nthreads, buffer_read_latency, 1, ports, ChirrtlMemoryReadUnderWrite::Undefined),
        Some(TypeTree::build_from_type(&mem_type, TypeDirection::Outgoing))
    );
    fg.graph.add_node(mem_node)
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
    let read_latency = memport.read_latency;
    let host_clock_id = memport.host_clock_id;
    let host_clock = memport.host_clock;

    if read_latency == 0 {
        // If read latency is zero, just connect the rdata port to the sinks
        for eid in find_memport_outgoing_edges(fg, memport, &rdata) {
            let ep = fg.graph.edge_endpoints(eid).unwrap();
            let mut edge = fg.graph.edge_weight(eid).unwrap().clone();
            edge.src = Expr::Reference(memport.field_ref(&rdata));
            fg.graph.add_edge(memport.new_id, ep.1, edge);
        }

        // Also connect to output node's bits field if it exists
        if let Some(output_id) = find_output_node_for_read_data(fg, memport) {
            let output_name = fg.graph.node_weight(output_id).unwrap().name.as_ref().unwrap().clone();
            let bits_ref = Reference::RefDot(
                Box::new(Reference::Ref(output_name)),
                Identifier::Name("bits".to_string())
            );
            let read_data_expr = Expr::Reference(memport.field_ref(&rdata));
            fg.add_wire(
                memport.new_id,
                read_data_expr,
                output_id,
                Some(bits_ref),
            );
        }
    } else {
        // Read latency > 0 case: Create buffer memory and edge detection logic
        // Based on ImplementThreadedMems.scala

        // If there are at least 4 threads, we can extend the latency of read data propagation
        let bram_to_buffer_pipe_depth: u32 = if nthreads < 4 { 0 } else { 1 };
        let buffer_read_latency: u32 = if nthreads < 4 { 0 } else { 1 };

        // Get port name as string for creating meaningful names
        let port_name_str = match memport.port {
            Identifier::Name(s) => s.clone(),
            _ => "port".to_string(),
        };

        // Get the memory's BASE data type (not the full memory type with ports)
        // This is the type of each element stored in the memory
        let mem_base_type = fg.memory_base_type(memport.new_id);
        let data_ttree = TypeTree::build_from_type(&mem_base_type, TypeDirection::Outgoing);

        // Create buffer memory name: {port}_datas for read ports, {port}_rdatas for readwrite ports
        let buffer_mem_suffix = if rdata == Identifier::Name("data".to_string()) { "datas" } else { "rdatas" };
        let buffer_mem_name = fg.namespace.new_name(&format!("{}_{}", port_name_str, buffer_mem_suffix));
        let buffer_mem_id = create_buffer_memory(fg, buffer_mem_name.clone(), &data_ttree, nthreads, buffer_read_latency);

        // Get thread_idx name and expression
        let thread_idx_name = fg.graph.node_weight(thread_idx_id).unwrap().name.as_ref().unwrap().clone();
        let thread_idx_expr = Expr::Reference(Reference::Ref(thread_idx_name.clone()));
        let thread_idx_bits = log2_ceil(nthreads);

        // Find target clock from the memory port's clk connection
        // The target clock is the clock connected to the original memory port
        let clk_field = Identifier::Name("clk".to_string());
        let target_clock_eid = find_memport_incoming_edge(fg, memport, &clk_field);

        let (target_clock_id, target_clock_expr) = if let Some(eid) = target_clock_eid {
            let ep = fg.graph.edge_endpoints(eid).unwrap();
            let edge = fg.graph.edge_weight(eid).unwrap();
            (ep.0, edge.src.clone())
        } else {
            unreachable!()
        };

        // Create target clock counter - a register that toggles on each target clock edge
        // This is used for edge detection
        let edge_count_name = fg.namespace.new_name("edgeCount");
        let edge_count_ttree = TypeTree::build_from_ground_type(GroundType::UInt(Some(Width(1))));
        let edge_count_node = FirNode::new(
            Some(edge_count_name.clone()),
            FirNodeType::Reg,
            Some(edge_count_ttree)
        );
        let edge_count_id = fg.graph.add_node(edge_count_node);

        // Connect target clock to edge counter
        let target_clock_edge = FirEdge::new(
            target_clock_expr.clone(),
            None,
            FirEdgeType::Clock
        );
        fg.graph.add_edge(target_clock_id, edge_count_id, target_clock_edge);

        // Create NOT gate for edge counter update (toggle)
        let edge_count_ref = Expr::Reference(Reference::Ref(edge_count_name.clone()));
        let not_name = fg.namespace.new_name("edgeCount_not");
        let not_node = FirNode::new(
            Some(not_name.clone()),
            FirNodeType::PrimOp1Expr(PrimOp1Expr::Not),
            None
        );
        let not_id = fg.graph.add_node(not_node);
        let not_input_edge = FirEdge::new(
            edge_count_ref.clone(),
            None,
            FirEdgeType::Operand0
        );
        fg.graph.add_edge(edge_count_id, not_id, not_input_edge);

        // Connect NOT output to edge counter register
        let not_expr = Expr::Reference(Reference::Ref(not_name));
        let edge_update_edge = FirEdge::new(
            not_expr,
            Some(Reference::Ref(edge_count_name.clone())),
            FirEdgeType::Wire
        );
        fg.graph.add_edge(not_id, edge_count_id, edge_update_edge);

        // Create edge count tracker - samples edge count with host clock
        let tracker_name = fg.namespace.new_name("edgeCountTracker");
        let (tracker_id, tracker_expr) = host_reg_next(
            fg,
            tracker_name,
            edge_count_id,
            edge_count_ref.clone(),
            host_clock_id,
            host_clock,
            1
        );

        // Create edge status: XOR of tracker and edge count (detects edges)
        let edge_status_name = fg.namespace.new_name("edgeStatus");
        let (edge_status_id, edge_status_expr) = fg.add_primop2_with_name(
            edge_status_name,
            PrimOp2Expr::Xor,
            tracker_id,
            tracker_expr,
            edge_count_id,
            edge_count_ref
        );

        let (tidx_last_id, tidx_last_expr) = pipeline(
            fg,
            1,
            thread_idx_id,
            thread_idx_expr.clone(),
            "tIdxLast",
            host_clock_id,
            host_clock,
            thread_idx_bits
        );

        let (tidx_piped_id, tidx_piped_expr) = pipeline(
            fg,
            bram_to_buffer_pipe_depth + buffer_read_latency + 1,
            tidx_last_id,
            tidx_last_expr.clone(),
            "tIdxPiped",
            host_clock_id,
            host_clock,
            thread_idx_bits
        );

        let (edge_status_piped_id, edge_status_piped_expr) = pipeline(
            fg,
            bram_to_buffer_pipe_depth,
            edge_status_id,
            edge_status_expr,
            "edgeStatusPiped",
            host_clock_id,
            host_clock,
            1
        );

        // Create node for main memory's read data output
        // This captures the data coming out of the main memory
        // Name format: {port}_{rdata}_node (e.g., "reads_0_data_node")
        let rdata_str = match &rdata {
            Identifier::Name(s) => s.clone(),
            _ => "data".to_string(),
        };
        let dout_node_name = fg.namespace.new_name(&format!("{}_{}_node", port_name_str, rdata_str));
        let dout_node = FirNode::new(
            Some(dout_node_name.clone()),
            FirNodeType::Wire,
            Some(data_ttree.clone())
        );
        let dout_node_id = fg.graph.add_node(dout_node);

        // Connect main memory's read data to the node
        let main_mem_rdata_expr = Expr::Reference(memport.field_ref(&rdata));
        let dout_edge = FirEdge::new(
            main_mem_rdata_expr,
            Some(Reference::Ref(dout_node_name.clone())),
            FirEdgeType::Wire
        );
        fg.graph.add_edge(memport.new_id, dout_node_id, dout_edge);

        // Pipeline the data output by bramToBufferPipeDepth
        // For data pipelines, we use 64-bit width as default since the exact width doesn't 
        // affect correctness (only some synthesis optimizations)
        let data_width = 64;
        let dout_node_expr = Expr::Reference(Reference::Ref(dout_node_name));
        let (dout_piped_id, dout_piped_expr) = pipeline(
            fg,
            bram_to_buffer_pipe_depth,
            dout_node_id,
            dout_node_expr,
            &format!("{}_{}_piped", port_name_str, rdata_str),
            host_clock_id,
            host_clock,
            data_width
        );

        // Helper to create buffer memory port references
        let buffer_rport_ref = |field: &str| -> Reference {
            Reference::RefDot(
                Box::new(Reference::RefDot(
                    Box::new(Reference::Ref(buffer_mem_name.clone())),
                    Identifier::Name("r".to_string())
                )),
                Identifier::Name(field.to_string())
            )
        };
        let buffer_wport_ref = |field: &str| -> Reference {
            Reference::RefDot(
                Box::new(Reference::RefDot(
                    Box::new(Reference::Ref(buffer_mem_name.clone())),
                    Identifier::Name("w".to_string())
                )),
                Identifier::Name(field.to_string())
            )
        };

        let host_clock_expr = Expr::Reference(Reference::Ref(host_clock.clone()));
        fg.add_wire(host_clock_id, host_clock_expr.clone(), buffer_mem_id, Some(buffer_rport_ref("clk")));

        fg.add_wire(tidx_last_id, tidx_last_expr, buffer_mem_id, Some(buffer_rport_ref("addr")));

        let (one_id, one_expr) = fg.add_uint_literal(1, 1);
        fg.add_wire(one_id, one_expr, buffer_mem_id, Some(buffer_rport_ref("en")));

        fg.add_wire(host_clock_id, host_clock_expr, buffer_mem_id, Some(buffer_wport_ref("clk")));

        fg.add_wire(tidx_piped_id, tidx_piped_expr, buffer_mem_id, Some(buffer_wport_ref("addr")));

        fg.add_wire(edge_status_piped_id, edge_status_piped_expr, buffer_mem_id, Some(buffer_wport_ref("en")));

        let (mask_one_id, mask_one_expr) = fg.add_uint_literal(1, 1);
        fg.add_wire(mask_one_id, mask_one_expr, buffer_mem_id, Some(buffer_wport_ref("mask")));

        fg.add_wire(dout_piped_id, dout_piped_expr, buffer_mem_id, Some(buffer_wport_ref("data")));

        let buffer_rdata_expr = Expr::Reference(buffer_rport_ref("data"));

        for eid in find_memport_outgoing_edges(fg, memport, &rdata) {
            let ep = fg.graph.edge_endpoints(eid).unwrap();
            let mut edge = fg.graph.edge_weight(eid).unwrap().clone();
            edge.src = buffer_rdata_expr.clone();
            fg.graph.add_edge(buffer_mem_id, ep.1, edge);
        }

        if let Some(output_id) = find_output_node_for_read_data(fg, memport) {
            let output_name = fg.graph.node_weight(output_id).unwrap().name.as_ref().unwrap().clone();
            let bits_ref = Reference::RefDot(
                Box::new(Reference::Ref(output_name)),
                Identifier::Name("bits".to_string())
            );
            fg.add_wire(
                buffer_mem_id,
                buffer_rdata_expr,
                output_id,
                Some(bits_ref),
            );
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
