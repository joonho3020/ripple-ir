use crate::ir::whentree::WhenTree;
use crate::ir::typetree::{TypeDirection, TypeTree};
use crate::ir::firir::*;
use crate::ir::PhiPriority;
use crate::passes::check_ast_assumption::*;
use chirrtl_parser::ast::*;
use petgraph::graph::NodeIndex;
use indexmap::{IndexMap, IndexSet};
use petgraph::visit::EdgeRef;
use petgraph::Direction::Incoming;

type RefMap = IndexMap<Reference, NodeIndex>;
type RefSet = IndexSet<Reference>;

/// Book-keeping datastructure required when converting the FIRRTL AST to graph form
#[derive(Default, Debug)]
struct NodeMap {
    /// `Reference` (in the FIRRTL AST) -> `NodeIndex` of the graph
    pub node_map: RefMap,

    /// `Reference` (sink node of the phi node) -> `NodeIndex` of the phi node
    pub phi_map: RefMap,

    /// The if key exists, phi node has been connected to its sink
    pub phi_connected: RefSet,
}

/// Create a graph based IR from the FIRRTL AST
pub fn from_circuit(ast: &Circuit) -> FirIR {
    let mut ret = FirIR::new(ast.name.clone());
    for module in ast.modules.iter() {
        let (name, ripple_graph) = from_circuit_module(module);
        ret.graphs.insert(name.clone(), ripple_graph);
    }
    return ret;
}

pub fn from_circuit_module(module: &CircuitModule) -> (&Identifier, FirGraph) {
    let mut nodemap: NodeMap = NodeMap::default();
    match module {
        CircuitModule::Module(m) => {
            (&m.name, from_module(m, &mut nodemap))
        }
        CircuitModule::ExtModule(e) => {
            (&e.name, from_ext_module(e, &mut nodemap))
        }
    }
}

fn from_module(module: &Module, nm: &mut NodeMap) -> FirGraph {
    let mut ret = FirGraph::new();
    collect_graph_nodes_from_ports(&mut ret, &module.ports, nm);
    collect_graph_nodes_from_stmts(&mut ret, &module.stmts, nm);
    connect_graph_edges_from_stmts(&mut ret, &module.stmts, nm);
    connect_phi_in_edges_from_stmts(&mut ret, &module.stmts, nm);
    connect_phi_node_sels(&mut ret, nm);
    return ret;
}

fn from_ext_module(module: &ExtModule, nm: &mut NodeMap) -> FirGraph {
    let ret = FirGraph::new();
    return ret;
}

/// Create graph nodes from the ports of this module
fn collect_graph_nodes_from_ports(ir: &mut FirGraph, ports: &Ports, nm: &mut NodeMap) {
    for port in ports.iter() {
        match port.as_ref() {
            Port::Input(name, tpe, _info) => {
                let typetree = TypeTree::build_from_type(tpe, TypeDirection::Outgoing);
                let all_refs = typetree.all_possible_references(name.clone());
                let id = ir.graph.add_node(
                    FirNode::new(
                        Some(name.clone()),
                        FirNodeType::Input,
                        Some(typetree)
                    ));

                // Add all possible references in the `TypeTree` as downstream
                // references can use a subset of a AggregateType
                for reference in all_refs {
                    nm.node_map.insert(reference, id);
                }
            }
            Port::Output(name, tpe, _info) => {
                let typetree = TypeTree::build_from_type(tpe, TypeDirection::Incoming);
                let all_refs = typetree.all_possible_references(name.clone());
                let id = ir.graph.add_node(
                    FirNode::new(
                        Some(name.clone()),
                        FirNodeType::Output,
                        Some(typetree)
                    ));

                for reference in all_refs {
                    nm.node_map.insert(reference, id);
                }
            }
        }
    }
}

/// Create graph nodes from the module statements
fn collect_graph_nodes_from_stmts(ir: &mut FirGraph, stmts: &Stmts, nm: &mut NodeMap) {
    for stmt in stmts {
        add_graph_node_from_stmt(ir, stmt.as_ref(), nm);
    }
}

fn add_node(
    ir: &mut FirGraph,
    tpe_opt: Option<&Type>,
    name_opt: Option<Identifier>,
    dir: TypeDirection,
    nt: FirNodeType
) -> NodeIndex {
    let typetree_opt = match tpe_opt {
        Some(tpe) => Some(TypeTree::build_from_type(tpe, dir)),
        None => None,
    };
    return ir.graph.add_node(FirNode::new(name_opt, nt, typetree_opt));
}

/// Create a graph node for a statement
fn add_graph_node_from_stmt(ir: &mut FirGraph, stmt: &Stmt, nm: &mut NodeMap) {
    match stmt {
        Stmt::When(_cond, _info, when_stmts, else_stmts_opt) => {
            collect_graph_nodes_from_stmts(ir, when_stmts, nm);
            if let Some(else_stmts) = else_stmts_opt {
                collect_graph_nodes_from_stmts(ir, else_stmts, nm);
            }
        }
        Stmt::Wire(name, tpe, _info) => {
            let id = add_node(ir, Some(tpe), Some(name.clone()), TypeDirection::Outgoing, FirNodeType::Wire);
            nm.node_map.insert(Reference::Ref(name.clone()), id);
        }
        Stmt::Reg(name, tpe, _clk, _info) => {
            let id = add_node(ir, Some(tpe), Some(name.clone()), TypeDirection::Outgoing, FirNodeType::Reg);
            nm.node_map.insert(Reference::Ref(name.clone()), id);
        }
        Stmt::RegReset(name, tpe, _clk, _rst, _init, _info) => {
            let id = add_node(ir, Some(tpe), Some(name.clone()), TypeDirection::Outgoing, FirNodeType::RegReset);
            nm.node_map.insert(Reference::Ref(name.clone()), id);
        }
        Stmt::ChirrtlMemory(mem) => {
            match mem {
                ChirrtlMemory::SMem(name, tpe, ruw_opt, _info) => {
                    let id = add_node(ir, Some(tpe), Some(name.clone()), TypeDirection::Outgoing, FirNodeType::SMem(ruw_opt.clone()));
                    nm.node_map.insert(Reference::Ref(name.clone()), id);
                }
                ChirrtlMemory::CMem(name, tpe, _info) => {
                    let id = add_node(ir, Some(tpe), Some(name.clone()), TypeDirection::Outgoing, FirNodeType::CMem);
                    nm.node_map.insert(Reference::Ref(name.clone()), id);
                }
            }
        }
        Stmt::ChirrtlMemoryPort(mport) => {
            let (name, id) = match mport {
                ChirrtlMemoryPort::Write(port, _mem, _addr, _clk, _info) => {
                    let id = add_node(ir, None, Some(port.clone()), TypeDirection::Outgoing, FirNodeType::WriteMemPort);
                    (port, id)
                }
                ChirrtlMemoryPort::Read(port, _mem, _addr, _clk, _info) => {
                    let id = add_node(ir, None, Some(port.clone()), TypeDirection::Outgoing, FirNodeType::ReadMemPort);
                    (port, id)
                }
                ChirrtlMemoryPort::Infer(port, _mem, _addr, _clk, _info) => {
                    let id = add_node(ir, None, Some(port.clone()), TypeDirection::Outgoing, FirNodeType::InferMemPort);
                    (port, id)
                }
            };
            nm.node_map.insert(Reference::Ref(name.clone()), id);
        }
        Stmt::Inst(inst_name, mod_name, _info) => {
            let id = add_node(ir, None, Some(inst_name.clone()), TypeDirection::Outgoing, FirNodeType::Inst(mod_name.clone()));
            nm.node_map.insert(Reference::Ref(inst_name.clone()), id);
        }
        Stmt::Node(out_name, expr, _info) => {
            // We add the constant nodes (e.g., UInt<1>(0)) later when adding
            // edges as there is no way to reference constant nodes and place them
            // in `nm.node_map`
            match expr {
                Expr::Mux(_, _, _) => {
                    let id = add_node(ir, None, None, TypeDirection::Outgoing, FirNodeType::Mux);
                    nm.node_map.insert(Reference::Ref(out_name.clone()), id);
                }
                Expr::PrimOp2Expr(op, _, _) => {
                    let id = add_node(ir, None, Some(out_name.clone()), TypeDirection::Outgoing, FirNodeType::PrimOp2Expr(*op));
                    nm.node_map.insert(Reference::Ref(out_name.clone()), id);
                }
                Expr::PrimOp1Expr(op, _) => {
                    let id = add_node(ir, None, Some(out_name.clone()), TypeDirection::Outgoing, FirNodeType::PrimOp1Expr(*op));
                    nm.node_map.insert(Reference::Ref(out_name.clone()), id);
                }
                Expr::PrimOp1Expr1Int(op, _, x) => {
                    let id = add_node(ir, None, Some(out_name.clone()), TypeDirection::Outgoing, FirNodeType::PrimOp1Expr1Int(*op, x.clone()));
                    nm.node_map.insert(Reference::Ref(out_name.clone()), id);
                }
                Expr::PrimOp1Expr2Int(op, _, x, y) => {
                    let id = add_node(ir, None, Some(out_name.clone()), TypeDirection::Outgoing, FirNodeType::PrimOp1Expr2Int(*op, x.clone(), y.clone()));
                    nm.node_map.insert(Reference::Ref(out_name.clone()), id);
                }
                Expr::Reference(_)      |
                    Expr::UIntNoInit(_) |
                    Expr::UIntInit(_, _) |
                    Expr::SIntNoInit(_) |
                    Expr::SIntInit(_, _) |
                    Expr::ValidIf(_, _) => {
                        assert!(false, "Unexpected node right hand side {:?}", expr);
                    }
            }
        }
        Stmt::Connect(expr, _, _info) |
            Stmt::Invalidate(expr, _info) => {
            // Insert phi nodes for all possible lhs connections
            // We can remove the unnecessary ones later
            match expr {
                Expr::Reference(r) => {
                    let root_name = r.root();
                    let root_ref = Reference::Ref(root_name.clone());
                    if !nm.phi_map.contains_key(&root_ref) {
                        let id = add_node(ir, None, Some(root_name.clone()), TypeDirection::Outgoing, FirNodeType::Phi);
                        nm.phi_map.insert(root_ref, id);
                    }
                }
                _ => {
                    panic!("Connect/Invalidate stmt {:?} expr {:?} unhandled type", stmt, expr);
                }
            }
        }
        Stmt::Skip(..) => {
        }
        Stmt::Printf(_clk, _posedge, _msg, _exprs_opt, _info) => {
        }
        Stmt::Assert(_clk, _pred, _cond, _msg, _info) => {
        }
    }
}

/// Create graph edges from statements
fn connect_graph_edges_from_stmts(ir: &mut FirGraph, stmts: &Stmts, nm: &mut NodeMap) {
    for stmt in stmts {
        add_graph_edge_from_stmt(ir, stmt.as_ref(), nm);
    }
}

/// Recursively traverse the Reference in cases an Expr has been used to
/// index into an array reference
fn add_graph_edge_from_ref(
    ir: &mut FirGraph,
    dst_id: NodeIndex,
    reference: &Reference,
    edge_type: FirEdge,
    nm: &NodeMap
) {
    match reference {
        Reference::Ref(_) => {
            let src_id = nm.node_map.get(reference)
                .expect(&format!("Driver reference {:?} not found in node_map", reference));

            ir.graph.add_edge(*src_id, dst_id, edge_type);
        }
        Reference::RefDot(par, _) => {
            add_graph_edge_from_ref(ir, dst_id, par.as_ref(), edge_type, nm);
        }
        Reference::RefIdxInt(par, _) => {
            add_graph_edge_from_ref(ir, dst_id, par.as_ref(), edge_type, nm);
        }
        Reference::RefIdxExpr(par, expr) => {
            add_graph_edge_from_ref(ir, dst_id, par.as_ref(), edge_type, nm);

            let root_ref = Reference::Ref(par.root());
            let arr_id = nm.node_map.get(&root_ref)
                .expect(&format!("Array reference {:?} not found in node_map", par.as_ref()));

            let arr_addr_edge = FirEdge::new(expr.as_ref().clone(), None, FirEdgeType::ArrayAddr);
            add_graph_edge_from_expr(ir, *arr_id, &expr, arr_addr_edge, nm);
        }
    }
}

/// Given an expression (`rhs`), connect it to the sink node `dst_id` in the IR graph.
/// Creates constant nodes that weren't created in the `collect_graph_nodes_from_stmts` step
fn add_graph_edge_from_expr(
    ir: &mut FirGraph,
    dst_id: NodeIndex,
    rhs: &Expr,
    edge_type: FirEdge,
    nm: &NodeMap
) {
    match rhs {
        Expr::Reference(r) => {
            add_graph_edge_from_ref(ir, dst_id, r, edge_type, nm);
        }
        Expr::UIntInit(w, init) => {
            let src_id = add_node(ir, None, None, TypeDirection::Outgoing, FirNodeType::UIntLiteral(*w, init.clone()));
            ir.graph.add_edge(src_id, dst_id, edge_type);
        }
        Expr::SIntInit(w, init) => {
            let src_id = add_node(ir, None, None, TypeDirection::Outgoing, FirNodeType::SIntLiteral(*w, init.clone()));
            ir.graph.add_edge(src_id, dst_id, edge_type);
        }
        Expr::PrimOp1Expr(op, expr) => {
            let op_id = add_node(ir, None, None, TypeDirection::Outgoing, FirNodeType::PrimOp1Expr(*op));
            let src_id = match expr.as_ref() {
                Expr::UIntInit(w, init) => {
                    add_node(ir, None, None, TypeDirection::Outgoing, FirNodeType::UIntLiteral(*w, init.clone()))
                },
                Expr::SIntInit(w, init) => {
                    add_node(ir, None, None, TypeDirection::Outgoing, FirNodeType::SIntLiteral(*w, init.clone()))
                }
                _ => {
                    panic!("Connect rhs PrimOp1Expr expr not a const {:?}", rhs);
                }
            };
            let mut src_edge_type = edge_type.clone();
            src_edge_type.et = FirEdgeType::Operand0;
            ir.graph.add_edge(src_id, op_id, src_edge_type);
            ir.graph.add_edge(op_id, dst_id, edge_type);
        }
        _ => {
            assert!(false, "Connect rhs {:?} unhandled type", rhs);
        }
    }
}

/// Given a statement, adds a graph edge corresponding to the statement.
/// There are only a few cases that addes a edge between two nodes in a graph:
/// - `Reg`, `RegReset` clock, reset, init signals
/// - `Node` statements. These perform primitive operations, or muxes (combinational operations)
/// - `Connect`: phi nodes to their sink
///   - This won't connect the input and selection signals going into the phi nodes
/// - `ChirrtlMemoryPort`
///    - Connects the memory node with the port nodes
///    - Connects the port node to the clock node
///    - Connects the address node to the port node
fn add_graph_edge_from_stmt(ir: &mut FirGraph, stmt: &Stmt, nm: &mut NodeMap) {
    match stmt {
        Stmt::When(cond, _info, when_stmts, else_stmts_opt) => {
            connect_graph_edges_from_stmts(ir, when_stmts, nm);
            if let Some(else_stmts) = else_stmts_opt {
                connect_graph_edges_from_stmts(ir, else_stmts, nm);
            }
            assert!(is_reference_or_const(cond), "When condition should be a Reference {:?}", cond);
        }
        Stmt::Wire(..) |
            Stmt::Skip(..) => {
        }
        Stmt::Reg(name, _tpe, clk, _info) => {
            let reg_id = nm.node_map.get(&Reference::Ref(name.clone())).unwrap();
            if let Expr::Reference(_clk_ref) = clk {
                let edge = FirEdge::new(clk.clone(), None, FirEdgeType::Clock);
                add_graph_edge_from_expr(ir, *reg_id, clk, edge, nm);
            } else {
                panic!("clk {:?} for reg {:?} is not a reference", clk, name);
            }
        }
        Stmt::RegReset(name, _tpe, clk, rst, init, _info) => {
            match (clk, rst) {
                (Expr::Reference(_clk_ref), Expr::Reference(_rst_ref)) => {
                    let reg_ref = Reference::Ref(name.clone());
                    let reg_id = nm.node_map.get(&reg_ref).unwrap();

                    let clk_edge = FirEdge::new(clk.clone(), None, FirEdgeType::Clock);
                    let rst_edge = FirEdge::new(rst.clone(), None, FirEdgeType::Reset);
                    let init_edge = FirEdge::new(init.clone(), None, FirEdgeType::Wire);

                    add_graph_edge_from_expr(ir, *reg_id, clk,  clk_edge, nm);
                    add_graph_edge_from_expr(ir, *reg_id, rst,  rst_edge, nm);
                    add_graph_edge_from_expr(ir, *reg_id, init, init_edge, nm);
                }
                _ => {
                    panic!("No matching clk {:?} rst {:?} init {:?} for reg {:?}",
                        clk, rst, init, name);
                }
            }
        }
        Stmt::ChirrtlMemory(_mem) => {
        }
        Stmt::ChirrtlMemoryPort(mport) => {
            match mport {
                ChirrtlMemoryPort::Write(port, mem, addr, clk, _info) |
                ChirrtlMemoryPort::Read (port, mem, addr, clk, _info) |
                ChirrtlMemoryPort::Infer(port, mem, addr, clk, _info) => {
                    let port_ref = Reference::Ref(port.clone());
                    let mem_ref  = Reference::Ref(mem.clone());

                    let port_id = nm.node_map.get(&port_ref).unwrap();
                    let mem_id  = nm.node_map.get(&mem_ref).unwrap();

                    let port_expr = Expr::Reference(port_ref);
                    let mem_expr = Expr::Reference(mem_ref);
                    let clk_expr  = Expr::Reference(clk.clone());

                    let port_edge = FirEdge::new(port_expr.clone(), None, FirEdgeType::MemPortEdge);
                    let clk_edge  = FirEdge::new(clk_expr.clone(),  None, FirEdgeType::Clock);
                    let addr_edge = FirEdge::new(addr.clone(),      None, FirEdgeType::MemPortAddr);

                    add_graph_edge_from_expr(ir, *port_id,  &mem_expr, port_edge, nm);
                    add_graph_edge_from_expr(ir, *port_id, &clk_expr,  clk_edge, nm);
                    add_graph_edge_from_expr(ir, *port_id, addr, addr_edge, nm);
                }
            }
        }
        Stmt::Inst(_inst_name, _mod_name, _info) => {
        }
        Stmt::Node(out_name, expr, _info) => {
            let dst_id = nm.node_map.get(&Reference::Ref(out_name.clone()))
                .expect(&format!("Node lhs {:?} not found in node_map", stmt));

            match expr {
                Expr::Mux(cond, true_expr, false_expr) => {
                    let cond_edge = FirEdge::new(cond.as_ref().clone(), None, FirEdgeType::MuxCond);
                    let true_edge = FirEdge::new(true_expr.as_ref().clone(), None, FirEdgeType::MuxTrue);
                    let false_edge = FirEdge::new(false_expr.as_ref().clone(), None, FirEdgeType::MuxFalse);
                    add_graph_edge_from_expr(ir, *dst_id, cond, cond_edge, nm);
                    add_graph_edge_from_expr(ir, *dst_id, &true_expr, true_edge, nm);
                    add_graph_edge_from_expr(ir, *dst_id, &false_expr, false_edge, nm);
                }
                Expr::PrimOp2Expr(_, a, b) => {
                    let op0_edge = FirEdge::new(a.as_ref().clone(), None, FirEdgeType::Operand0);
                    let op1_edge = FirEdge::new(b.as_ref().clone(), None, FirEdgeType::Operand1);
                    add_graph_edge_from_expr(ir, *dst_id, &a, op0_edge, nm);
                    add_graph_edge_from_expr(ir, *dst_id, &b, op1_edge, nm);
                }
                Expr::PrimOp1Expr(_, a)            |
                    Expr::PrimOp1Expr1Int(_, a, _) |
                    Expr::PrimOp1Expr2Int(_, a, _, _) => {
                    let op0_edge = FirEdge::new(a.as_ref().clone(), None, FirEdgeType::Operand0);
                    add_graph_edge_from_expr(ir, *dst_id, &a, op0_edge, nm);
                }
                Expr::Reference(_)      |
                    Expr::UIntNoInit(_) |
                    Expr::UIntInit(_, _) |
                    Expr::SIntNoInit(_) |
                    Expr::SIntInit(_, _) |
                    Expr::ValidIf(_, _) => {
                        assert!(false, "Node rhs {:?} unhandled type", expr);
                }
            }
        }
        Stmt::Connect(lhs, _rhs, _info) => {
            let (dst_id, root_name) = match lhs {
                Expr::Reference(r) => {
                    (
                        nm.node_map.get(&Reference::Ref(r.root()))
                            .expect(&format!("Connect lhs not found in {:?}", lhs)),
                        r.root()
                    )
                }
                _ => {
                    panic!("Connect lhs {:?} unhandled type", lhs);
                }
            };

            // If phi node not yet connected, connect now
            let root_ref = Reference::Ref(root_name);
            if !nm.phi_connected.contains(&root_ref) {
                let phi_id = nm.phi_map.get(&root_ref)
                    .expect(&format!("Phi node for {:?} doesn't exist", root_ref));

                let root_expr = Expr::Reference(root_ref.clone());
                let phiout_edge = FirEdge::new(root_expr, None, FirEdgeType::PhiOut);
                ir.graph.add_edge(*phi_id, *dst_id, phiout_edge);
                nm.phi_connected.insert(root_ref);
            }
        }
        Stmt::Invalidate(expr, _info) => {
            if let Expr::Reference(r) = expr {
                let root_name = r.root();
                // If phi node not yet connected, connect now
                let root_ref = Reference::Ref(root_name);
                let phi_id = nm.phi_map.get(&root_ref)
                    .expect(&format!("Phi node for {:?} doesn't exist", root_ref));

                let dont_care_id = add_node(ir, None, None, TypeDirection::Outgoing, FirNodeType::DontCare);
                let dont_care_edge = FirEdge::new(expr.clone(), None, FirEdgeType::DontCare);
                ir.graph.add_edge(dont_care_id, *phi_id, dont_care_edge);
            } else {
                panic!("Invalidate expr {:?} is not a reference", expr);
            }
        }
        Stmt::Printf(_clk, _posedge, _msg, _exprs_opt, _info) => {
        }
        Stmt::Assert(_clk, _pred, _cond, _msg, _info) => {
        }
    }
}

/// Connect input signals going into the phi node
fn connect_phi_in_edges_from_stmts(ir: &mut FirGraph, stmts: &Stmts, nm: &mut NodeMap) {
    let mut whentree = WhenTree::new();
    whentree.from_stmts(stmts);
    let leaves = whentree.leaf_nodes();

    let mut block_prior_set: IndexSet<u32> = IndexSet::new();
    for leaf in leaves {
        let block_prior = leaf.priority;

        if block_prior_set.contains(&block_prior) {
            panic!("When block prior {} overlaps {:?}", block_prior, block_prior_set);
        }
        block_prior_set.insert(block_prior);

        for (stmt_prior, stmt) in leaf.stmts.iter().enumerate() {
            match stmt.as_ref() {
                Stmt::Connect(lhs, rhs, _info) => {
                    match lhs {
                        Expr::Reference(r) => {
                            let root_name = r.root();
                            let root_ref = Reference::Ref(root_name.clone());
                            let phi_id = nm.phi_map.get(&root_ref)
                                .expect(&format!("phi node for {:?} not found", root_ref));

                            let prior = PhiPriority::new(block_prior, stmt_prior as u32);
                            let edge = FirEdge::new(rhs.clone(), Some(r.clone()), FirEdgeType::PhiInput(prior, leaf.cond.clone()));
                            add_graph_edge_from_expr(ir, *phi_id, rhs, edge, nm);
                        }
                        _ => {
                            panic!("Connect lhs {:?} unhandled type", lhs);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

/// Given a phi node with `id`, collect all the select signals used by
/// the phi node inputs, and connect all these signals to the node.
fn connect_phi_node_sel_id(ir: &mut FirGraph, id: NodeIndex, nm: &mut NodeMap) {
    let mut sel_exprs: IndexSet<Expr> = IndexSet::new();

    let pedges = ir.graph.edges_directed(id, Incoming);
    for pedge_ref in pedges {
        let edge_w = ir.graph.edge_weight(pedge_ref.id()).unwrap();
        match &edge_w.et {
            FirEdgeType::PhiInput(_, cond) => {
                let sels = cond.collect_sels();
                for sel in sels {
                    sel_exprs.insert(sel);
                }
            }
            FirEdgeType::DontCare => {
                continue;
            }
            _ => {
                panic!("Unexpected Phi node driver edge {:?}", edge_w);
            }
        }
    }

    for sel in sel_exprs {
        let sel_edge = FirEdge::new(sel.clone(), None, FirEdgeType::PhiSel);
        add_graph_edge_from_expr(ir, id, &sel, sel_edge, nm);
    }
}

/// Connect the selection signals going into the phi nodes
fn connect_phi_node_sels(ir: &mut FirGraph, nm: &mut NodeMap) {
    for id in ir.graph.node_indices() {
        let node = ir.graph.node_weight(id).unwrap();
        match node.nt {
            FirNodeType::Phi => {
                connect_phi_node_sel_id(ir, id, nm);
            }
            _ => {
                continue;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::common::graphviz::GraphViz;
    use crate::common::RippleIRErr;
    use chirrtl_parser::parse_circuit;

    /// Run the AST to graph conversion and export the graph form
    fn run(name: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(format!("./test-inputs/{}.fir", name))?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let ir = from_circuit(&circuit);
        for (sub_name, graph) in ir.graphs {
            graph.export_graphviz(&format!("./test-outputs/{}-{}.dot.pdf", name, sub_name), None, true)?;
        }
        Ok(())
    }

    #[test]
    fn gcd() {
        run("GCD").expect("GCD");
    }

    #[test]
    fn nestedwhen() {
        run("NestedWhen").expect("NestedWhen");
    }

    #[test]
    fn nestedbundle() {
        run("NestedBundle").expect("NestedBundle");
    }

    #[test]
    fn nestedindex() {
        run("NestedIndex").expect("NestedIndex");
    }

    #[test]
    fn singleport_sram() {
        run("SinglePortSRAM").expect("SinglePortSRAM");
    }

    #[test]
    fn hierarchy() {
        run("Hierarchy").expect("Hierarchy");
    }

    use chirrtl_parser::lexer::FIRRTLLexer;
    use chirrtl_parser::firrtl::CircuitModuleParser;

    /// Check of the AST to graph conversion works for each CircuitModule
    fn run_check_completion(input_dir: &str) -> Result<(), RippleIRErr> {
        for entry in std::fs::read_dir(input_dir)? {
            let entry = entry?;
            let path = entry.path();

            // Check if it's a file (not a directory)
            if path.is_file() {
                match std::fs::read_to_string(&path) {
                    Ok(source) => {
                        let lexer = FIRRTLLexer::new(&source);
                        let parser = CircuitModuleParser::new();

                        println!("Parsing file: {:?}", path);
                        let ast = parser.parse(lexer).expect("TOWORK");
                        from_circuit_module(&ast);
                    }
                    Err(e) => {
                        eprintln!("Could not read file {}: {}", path.display(), e);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn rocket_check_completion() {
        run_check_completion("./test-inputs/rocket-modules/")
            .expect("rocket conversion failed");
    }

    #[test]
    fn boom_check_completion() {
        run_check_completion("./test-inputs/boom-modules/")
            .expect("boom conversion failed");
    }

    fn boom_module(name: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(format!("./test-inputs/boom-modules/{}.fir", name))?;
        let lexer = FIRRTLLexer::new(&source);
        let parser = CircuitModuleParser::new();

        let ast = parser.parse(lexer).expect("TOWORK");
        let (_, graph) = from_circuit_module(&ast);
        let dot = graph.graphviz_string(None)?;
// println!("{}", dot);

        Ok(())
    }

    #[test]
    fn boom_fetch_target_queue() {
        // Indirect addressing
        boom_module("FetchTargetQueue").expect("FetchTargetQueue");
    }

    #[test]
    fn boom_lsu() {
        // Indirect addressing
        boom_module("LSU").expect("LSU");
    }
}
