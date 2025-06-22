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
        let mem_type = node.ttree.as_ref().unwrap().to_type();

        // Create new memory with capacity multiplied by nthreads
        let new_depth = depth * nthreads;
        let new_mem_type = memory_type_from_ports(ports, new_depth, &mem_type);
        let new_mem = FirNode::new(
            Some(mem_name.clone()),
            FirNodeType::Memory(new_depth, *rlat, *wlat, ports.clone(), ruw.clone()),
            Some(TypeTree::build_from_type(&new_mem_type, TypeDirection::Outgoing))
        );

        let new_mem_id = fame5.graph.add_node(new_mem);

        for port in ports {
            match port.as_ref() {
                MemoryPort::Read(name) => {
                    let mport = MemPortInfo::new(mem_id, new_mem_id, &mem_name, name);
                    rport_connections(fame5, &mport, thread_idx_id, nthreads);
                },
                MemoryPort::Write(port_name) => {
                    let mport = MemPortInfo::new(mem_id, new_mem_id, &mem_name, port_name);
                    wport_connections(fame5, &mport, thread_idx_id);
                },
                MemoryPort::ReadWrite(port_name) => {
                    let mport = MemPortInfo::new(mem_id, new_mem_id, &mem_name, port_name);
                    rwport_connections(fame5, &mport, thread_idx_id, nthreads);
                },
            }
        }
    }
}

struct MemPortInfo<'a> {
    orig_id: NodeIndex,
    new_id: NodeIndex,
    mem: &'a Identifier,
    port: &'a Identifier
}

impl <'a> MemPortInfo<'a> {
    fn new(orig_id: NodeIndex, new_id: NodeIndex, mem: &'a Identifier, port: &'a Identifier) -> Self {
        Self { orig_id, new_id, mem, port }
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

fn add_input_addr_edge<'a>(
    fg: &mut FirGraph,
    thread_idx_id: NodeIndex,
    memport: &'a MemPortInfo
)
{
    let addr = Identifier::Name("addr".to_string());
    let addr_driver = find_memport_incoming_edge(fg, memport, &addr);
    if let Some(addr_eid) = addr_driver {
        let ep = fg.graph.edge_endpoints(addr_eid).unwrap();
        let addr_driver_id = ep.0;
        let addr_driver_expr = fg.graph.edge_weight(addr_eid).unwrap().src.clone();

        // Get thread_idx name from the node
        let thread_idx_name = fg.graph.node_weight(thread_idx_id).unwrap().name.as_ref().unwrap().clone();
        let thread_idx_expr = Expr::Reference(Reference::Ref(thread_idx_name));

        // Create concatenation node (thread_idx concatenated with original address)
        let (cat_id, cat_expr) = fg.add_primop2(
            PrimOp2Expr::Cat,
            thread_idx_id,
            thread_idx_expr,
            addr_driver_id,
            addr_driver_expr,
        );

        // Connect concatenated address to the new memory port
        fg.add_wire(
            cat_id,
            cat_expr,
            memport.new_id,
            Some(memport.field_ref(&addr)),
        );
    }
}

fn add_output_rdata_edge<'a>(
    fg: &mut FirGraph,
    rdata: Identifier,
    memport: &'a MemPortInfo,
    thread_idx_id: NodeIndex,
    nthreads: u32,
)
{
    // Get the original memory node to check read latency
    let orig_mem_node = fg.graph.node_weight(memport.orig_id).unwrap();
    let read_latency = if let FirNodeType::Memory(_, rlat, _, _, _) = orig_mem_node.nt {
        rlat
    } else {
        panic!("Expected memory node");
    };

    if read_latency == 0 {
        // If read latency is zero, just connect the rdata port to the sinks
        for eid in find_memport_outgoing_edges(fg, memport, &rdata) {
            let ep = fg.graph.edge_endpoints(eid).unwrap();
            let mut edge = fg.graph.edge_weight(eid).unwrap().clone();
            edge.src = Expr::Reference(memport.field_ref(&rdata));
            fg.graph.add_edge(memport.new_id, ep.1, edge);
        }
    } else {
        // If read latency is one, create a combinational memory
        let rdata = Identifier::Name("data".to_string());

        // Create combinational memory with depth nthreads
        let rport_name = Identifier::Name("rd".to_string());
        let wport_name = Identifier::Name("wr".to_string());
        let ports = vec![
            Box::new(MemoryPort::Read(rport_name.clone())),
            Box::new(MemoryPort::Write(wport_name.clone())),
        ];

        // Get the data type from the original memory
        let mem_type = orig_mem_node.ttree.as_ref().unwrap().to_type();
        let comb_mem_type = memory_type_from_ports(&ports, nthreads, &mem_type);
        let comb_mem_name = Identifier::Name(format!("{}_fame5_pipeline", memport.mem));
        let comb_mem = FirNode::new(
            Some(comb_mem_name.clone()),
            FirNodeType::Memory(nthreads, 0, 1, ports, ChirrtlMemoryReadUnderWrite::Undefined),
            Some(TypeTree::build_from_type(&comb_mem_type, TypeDirection::Outgoing))
        );

        let comb_mem_id = fg.graph.add_node(comb_mem);

        // Create references for the combinational memory ports
        let rport_ref = Reference::RefDot(
            Box::new(Reference::Ref(comb_mem_name.clone())),
            rport_name.clone()
        );
        let wport_ref = Reference::RefDot(
            Box::new(Reference::Ref(comb_mem_name.clone())),
            wport_name.clone()
        );

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
            Some(Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("mask".to_string()))),
        );

        // Connect write port clock (same as original read port clock)
        if let Some(clk_eid) = find_memport_incoming_edge(fg, memport, &Identifier::Name("clk".to_string())) {
            let ep = fg.graph.edge_endpoints(clk_eid).unwrap();
            let clk_edge = fg.graph.edge_weight(clk_eid).unwrap().clone();
            let mut new_clk_edge = clk_edge.clone();
            new_clk_edge.dst = Some(Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("clk".to_string())));
            fg.graph.add_edge(ep.0, comb_mem_id, new_clk_edge);
        }

        // Connect write port address: thread_idx - 1 (if thread_idx == 0, then nthreads - 1)
        let thread_idx_bits = log2_ceil(nthreads);

        // Create comparison node to check if thread_idx == 0
        let (zero_const_id, zero_expr) = fg.add_uint_literal(0, thread_idx_bits);
        let (eq_id, eq_expr) = fg.add_primop2(
            PrimOp2Expr::Eq,
            thread_idx_id,
            Expr::Reference(Reference::Ref(thread_idx_name.clone())),
            zero_const_id,
            zero_expr,
        );

        // Create subtract node for thread_idx - 1
        let (one_const_id, one_expr) = fg.add_uint_literal(1, thread_idx_bits);
        let (sub_id, sub_expr) = fg.add_primop2(
            PrimOp2Expr::Sub,
            thread_idx_id,
            Expr::Reference(Reference::Ref(thread_idx_name.clone())),
            one_const_id,
            one_expr,
        );

        // Create constant for nthreads - 1
        let (nthreads_minus_one_id, nthreads_minus_one_expr) = fg.add_uint_literal(nthreads - 1, thread_idx_bits);

        // Create mux to select between thread_idx - 1 and nthreads - 1
        let (mux_id, mux_expr) = fg.add_mux(
            eq_id,
            eq_expr,
            nthreads_minus_one_id,
            nthreads_minus_one_expr,
            sub_id,
            sub_expr,
        );

        // Connect mux output to write port address
        fg.add_wire(
            mux_id,
            mux_expr,
            comb_mem_id,
            Some(Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("addr".to_string()))),
        );

        // Connect write port data from original memory rdata
        if let Some(rdata_eid) = find_memport_incoming_edge(fg, memport, &rdata) {
            let rdata_edge = fg.graph.edge_weight(rdata_eid).unwrap().clone();
            let mut new_rdata_edge = rdata_edge.clone();
            new_rdata_edge.dst = Some(Reference::RefDot(Box::new(wport_ref.clone()), Identifier::Name("data".to_string())));
            fg.graph.add_edge(memport.new_id, comb_mem_id, new_rdata_edge);
        }

        // Connect combinational memory read data to original sinks
        if let Some(eid) = find_memport_incoming_edge(fg, memport, &rdata) {
            let ep = fg.graph.edge_endpoints(eid).unwrap();
            let mut edge = fg.graph.edge_weight(eid).unwrap().clone();
            edge.src = Expr::Reference(Reference::RefDot(Box::new(rport_ref), Identifier::Name("data".to_string())));
            fg.graph.add_edge(comb_mem_id, ep.1, edge);
        }
    }
}

fn wport_connections<'a>(
    fg: &mut FirGraph,
    memport: &'a MemPortInfo,
    thread_idx_id: NodeIndex,
)
{
    add_input_addr_edge(fg, thread_idx_id, memport);
    add_input_mem_edge(fg, Identifier::Name("en".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("clk".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("data".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("mask".to_string()), memport);
}

fn rwport_connections<'a>(
    fg: &mut FirGraph,
    memport: &'a MemPortInfo,
    thread_idx_id: NodeIndex,
    nthreads: u32
)
{
    add_input_addr_edge(fg, thread_idx_id, memport);
    add_input_mem_edge(fg, Identifier::Name("en".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("clk".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("wmask".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("wdata".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("wmode".to_string()), memport);
    add_output_rdata_edge(fg, Identifier::Name("rdata".to_string()), memport, thread_idx_id, nthreads);
}

fn rport_connections<'a>(
    fg: &mut FirGraph,
    memport: &'a MemPortInfo,
    thread_idx_id: NodeIndex,
    nthreads: u32
)
{
    add_input_addr_edge(fg, thread_idx_id, memport);
    add_input_mem_edge(fg, Identifier::Name("en".to_string()), memport);
    add_input_mem_edge(fg, Identifier::Name("clk".to_string()), memport);
    add_output_rdata_edge(fg, Identifier::Name("data".to_string()), memport, thread_idx_id, nthreads);
}
