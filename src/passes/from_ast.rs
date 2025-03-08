use crate::parser::whentree::WhenTree;
use crate::{ir::*, parser::ast::*};
use crate::parser::typetree::Direction;
use petgraph::graph::NodeIndex;
use indexmap::{IndexMap, IndexSet};
use petgraph::visit::EdgeRef;
use petgraph::Direction::Incoming;

type RefMap = IndexMap<Reference, NodeIndex>;
type RefSet = IndexSet<Reference>;

#[derive(Default, Debug)]
struct NodeMap {
    /// `Reference` (in the FIRRTL AST) -> NodeIndex of the graph
    pub node_map: RefMap,

    /// `Reference` (sink node of the phi node) -> NodeIndex of the phi node
    pub phi_map: RefMap,

    /// The if key exists, phi node has been connected to its sink
    pub phi_connected: RefSet,
}

/// Create a graph based IR from the FIRRTL AST
pub fn from_circuit(ast: &Circuit) -> Vec<RippleIR> {
    let mut ret = vec![];
    for module in ast.modules.iter() {
        ret.push(from_circuit_module(module));
    }
    return ret;
}

fn from_circuit_module(module: &CircuitModule) -> RippleIR {
    let mut nodemap: NodeMap = NodeMap::default();
    match module {
        CircuitModule::Module(m) => {
            from_module(m, &mut nodemap)
        }
        CircuitModule::ExtModule(e) => {
            from_ext_module(e, &mut nodemap)
        }
    }
}

fn from_module(module: &Module, nm: &mut NodeMap) -> RippleIR {
    let mut ret = RippleIR::new();

    collect_graph_nodes_from_ports(&mut ret, &module.ports, nm);
    collect_graph_nodes_from_stmts(&mut ret, &module.stmts, nm);
    connect_graph_edges_from_stmts(&mut ret, &module.stmts, nm);
    connect_phi_in_edges_from_stmts(&mut ret, &module.stmts, nm);
    connect_phi_node_sels(&mut ret, nm);

    return ret;
}

fn from_ext_module(module: &ExtModule, nm: &mut NodeMap) -> RippleIR {
    let ret = RippleIR::new();
    return ret;
}

/// Returns all possible references of an `AggregateType`
/// - io: { a: UInt<1>, b: { c: UInt<1>, d: UInt<1> } }
/// - The above will return: io, io.a, io.b, io.b.c, io.b.c.d
fn all_references(tpe: &Type, name: &Identifier, dir: Option<Direction>) -> Vec<Reference> {
    let typetree = tpe.construct_tree(name.clone(), dir.unwrap_or(Direction::Output));
    typetree.all_possible_references()
}

/// Create graph nodes from the ports of this module
fn collect_graph_nodes_from_ports(ir: &mut RippleIR, ports: &Ports, nm: &mut NodeMap) {
    for port in ports.iter() {
        match port.as_ref() {
            Port::Input(name, tpe, _info) => {
                let id = ir.graph.add_node(NodeType::Input(name.clone(), tpe.clone()));

                // Add all possible references in the `TypeTree` as downstream
                // references can use a subset of a AggregateType
                let all_refs = all_references(tpe, name, Some(Direction::Input));
                for reference in all_refs {
                    nm.node_map.insert(reference, id);
                }
            }
            Port::Output(name, tpe, _info) => {
                let id = ir.graph.add_node(NodeType::Output(name.clone(), tpe.clone()));
                let all_refs = all_references(tpe, name, Some(Direction::Output));
                for reference in all_refs {
                    nm.node_map.insert(reference, id);
                }
            }
        }
    }
}

/// Create graph nodes from the module statements
fn collect_graph_nodes_from_stmts(ir: &mut RippleIR, stmts: &Stmts, nm: &mut NodeMap) {
    for stmt in stmts {
        add_graph_node_from_stmt(ir, stmt.as_ref(), nm);
    }
}

/// Create a graph node for a statement
fn add_graph_node_from_stmt(ir: &mut RippleIR, stmt: &Stmt, nm: &mut NodeMap) {
    match stmt {
        Stmt::When(_cond, _info, when_stmts, else_stmts_opt) => {
            collect_graph_nodes_from_stmts(ir, when_stmts, nm);
            if let Some(else_stmts) = else_stmts_opt {
                collect_graph_nodes_from_stmts(ir, else_stmts, nm);
            }
        }
        Stmt::Wire(name, tpe, _info) => {
            let id = ir.graph.add_node(NodeType::Wire(name.clone(), tpe.clone()));
            nm.node_map.insert(Reference::Ref(name.clone()), id);
        }
        Stmt::Reg(name, tpe, clk, _info) => {
            let id = ir.graph.add_node(NodeType::Reg(name.clone(), tpe.clone(), clk.clone()));
            nm.node_map.insert(Reference::Ref(name.clone()), id);
        }
        Stmt::RegReset(name, tpe, clk, rst, init, _info) => {
            let id = ir.graph.add_node(NodeType::RegReset(
                    name.clone(), tpe.clone(), clk.clone(), rst.clone(), init.clone()));
            nm.node_map.insert(Reference::Ref(name.clone()), id);
        }
        Stmt::ChirrtlMemory(_mem) => {
            todo!("ChirrtlMemory not yet implemented");
        }
        Stmt::ChirrtlMemoryPort(_mport) => {
            todo!("ChirrtlMemoryPort not yet implemented");
        }
        Stmt::Inst(_inst_name, _mod_name, _info) => {
            todo!("Instances not yet handled");
        }
        Stmt::Node(out_name, expr, _info) => {
            // We add the constant nodes (e.g., UInt<1>(0)) later when adding
            // edges as there is no way to reference constant nodes and place them
            // in `nm.node_map`
            match expr {
                Expr::Mux(_, _, _) => {
                    let id = ir.graph.add_node(NodeType::Mux);
                    nm.node_map.insert(Reference::Ref(out_name.clone()), id);
                }
                Expr::PrimOp2Expr(op, _, _) => {
                    let id = ir.graph.add_node(NodeType::PrimOp2Expr(*op));
                    nm.node_map.insert(Reference::Ref(out_name.clone()), id);
                }
                Expr::PrimOp1Expr(op, _) => {
                    let id = ir.graph.add_node(NodeType::PrimOp1Expr(*op));
                    nm.node_map.insert(Reference::Ref(out_name.clone()), id);
                }
                Expr::PrimOp1Expr1Int(op, _, x) => {
                    let id = ir.graph.add_node(NodeType::PrimOp1Expr1Int(*op, x.clone()));
                    nm.node_map.insert(Reference::Ref(out_name.clone()), id);
                }
                Expr::PrimOp1Expr2Int(op, _, x, y) => {
                    let id = ir.graph.add_node(NodeType::PrimOp1Expr2Int(*op, x.clone(), y.clone()));
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
        Stmt::Connect(lhs, _, _info) => {
            // Insert phi nodes for all possible lhs connections
            // We can remove the unnecessary ones later
            match lhs {
                Expr::Reference(r) => {
                    let root_name = r.root();
                    let root_ref = Reference::Ref(root_name.clone());
                    if !nm.phi_map.contains_key(&root_ref) {
                        let id = ir.graph.add_node(NodeType::Phi(root_name.clone()));
                        nm.phi_map.insert(root_ref, id);
                    }
                }
                _ => {
                    panic!("Connect lhs {:?} unhandled type", lhs);
                }
            }
        }
        Stmt::Invalidate(..) => {
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
fn connect_graph_edges_from_stmts(ir: &mut RippleIR, stmts: &Stmts, nm: &mut NodeMap) {
    for stmt in stmts {
        add_graph_edge_from_stmt(ir, stmt.as_ref(), nm);
    }
}

/// Given an expression (`rhs`), connect it to the sink node `dst_id` in the IR graph.
/// Creates constant nodes that weren't created in the `collect_graph_nodes_from_stmts` step
fn add_graph_edge_from_expr(
    ir: &mut RippleIR,
    dst_id: NodeIndex,
    rhs: &Expr,
    edge_type: EdgeType,
    nm: &NodeMap) {
    match rhs {
        Expr::Reference(r) => {
            let root_name = r.root();
            let src_id = nm.node_map.get(&Reference::Ref(root_name))
                .expect(&format!("Connect rhs {:?} not found in node_map", r));
            ir.graph.add_edge(*src_id, dst_id, edge_type);
        }
        Expr::UIntInit(w, init) => {
            let src_id = ir.graph.add_node(NodeType::UIntLiteral(*w, init.clone()));
            ir.graph.add_edge(src_id, dst_id, edge_type);
        }
        Expr::SIntInit(w, init) => {
            let src_id = ir.graph.add_node(NodeType::SIntLiteral(*w, init.clone()));
            ir.graph.add_edge(src_id, dst_id, edge_type);
        }
        Expr::PrimOp1Expr(op, expr) => {
            let op_id = ir.graph.add_node(NodeType::PrimOp1Expr(*op));
            let src_id = match expr.as_ref() {
                Expr::UIntInit(w, init) => {
                    ir.graph.add_node(NodeType::UIntLiteral(*w, init.clone()))
                },
                Expr::SIntInit(w, init) => {
                    ir.graph.add_node(NodeType::SIntLiteral(*w, init.clone()))
                }
                _ => {
                    panic!("Connect rhs PrimOp1Expr expr not a const {:?}", rhs);
                }
            };
            ir.graph.add_edge(op_id, dst_id, edge_type.clone());
            ir.graph.add_edge(src_id, op_id, edge_type);
        }
        _ => {
            assert!(false, "Connect rhs {:?} unhandled type", rhs);
        }
    }
}

/// Given a statement, add a graph edge
/// This won't connect the input and selection signals going into the phi nodes
fn add_graph_edge_from_stmt(ir: &mut RippleIR, stmt: &Stmt, nm: &mut NodeMap) {
    match stmt {
        Stmt::When(cond, _info, when_stmts, else_stmts_opt) => {
            connect_graph_edges_from_stmts(ir, when_stmts, nm);
            if let Some(else_stmts) = else_stmts_opt {
                connect_graph_edges_from_stmts(ir, else_stmts, nm);
            }
            assert!(is_reference_or_const(cond), "When condition should be a Reference {:?}", cond);
        }
        Stmt::Wire(..)           |
            Stmt::Reg(..)        |
            Stmt::RegReset(..)   |
            Stmt::Invalidate(..) |
            Stmt::Skip(..) => {
        }
        Stmt::ChirrtlMemory(_mem) => {
            todo!("ChirrtlMemory not yet handled");
        }
        Stmt::ChirrtlMemoryPort(_mport) => {
            todo!("ChirrtlMemoryPort not yet handled");
        }
        Stmt::Inst(_inst_name, _mod_name, _info) => {
            todo!("Instances not yet handled");
        }
        Stmt::Node(out_name, expr, _info) => {
            let dst_id = nm.node_map.get(&Reference::Ref(out_name.clone()))
                .expect(&format!("Node lhs {:?} not found in node_map", stmt));

            match expr {
                Expr::Mux(cond, true_expr, false_expr) => {
                    add_graph_edge_from_expr(ir, *dst_id, cond, EdgeType::Default, nm);
                    add_graph_edge_from_expr(ir, *dst_id, &true_expr, EdgeType::Default, nm);
                    add_graph_edge_from_expr(ir, *dst_id, &false_expr, EdgeType::Default, nm);
                }
                Expr::PrimOp2Expr(_, a, b) => {
                    add_graph_edge_from_expr(ir, *dst_id, &a, EdgeType::Default, nm);
                    add_graph_edge_from_expr(ir, *dst_id, &b, EdgeType::Default, nm);
                }
                Expr::PrimOp1Expr(_, a) => {
                    add_graph_edge_from_expr(ir, *dst_id, &a, EdgeType::Default, nm);
                }
                Expr::PrimOp1Expr1Int(_, a, _) => {
                    add_graph_edge_from_expr(ir, *dst_id, &a, EdgeType::Default, nm);
                }
                Expr::PrimOp1Expr2Int(_, a, _, _) => {
                    add_graph_edge_from_expr(ir, *dst_id, &a, EdgeType::Default, nm);
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

                ir.graph.add_edge(*phi_id, *dst_id, EdgeType::Default);
                nm.phi_connected.insert(root_ref);
            }
        }
        Stmt::Printf(_clk, _posedge, _msg, _exprs_opt, _info) => {
        }
        Stmt::Assert(_clk, _pred, _cond, _msg, _info) => {
        }
    }
}

/// Connect input signals going into the phi node
fn connect_phi_in_edges_from_stmts(ir: &mut RippleIR, stmts: &Stmts, nm: &mut NodeMap) {
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
                            let edge = EdgeType::PhiInput(prior, leaf.cond.clone());
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

fn connect_phi_node_sel_id(ir: &mut RippleIR, id: NodeIndex, nm: &mut NodeMap) {
    let mut sel_exprs: IndexSet<Expr> = IndexSet::new();

    let pedges = ir.graph.edges_directed(id, Incoming);
    for pedge_ref in pedges {
        let edge_w = ir.graph.edge_weight(pedge_ref.id()).unwrap();
        match edge_w {
            EdgeType::PhiInput(_, cond) => {
                let sels = cond.collect_sels();
                for sel in sels {
                    sel_exprs.insert(sel);
                }
            }
            _ => {
                panic!("Unexpected Phi node driver edge {:?}", edge_w);
            }
        }
    }

    for sel in sel_exprs {
        add_graph_edge_from_expr(ir, id, &sel, EdgeType::PhiSel(sel.clone()), nm);
    }
}

/// Connect the selection signals going into the phi nodes
fn connect_phi_node_sels(ir: &mut RippleIR, nm: &mut NodeMap) {
    for id in ir.graph.node_indices() {
        let node = ir.graph.node_weight(id).unwrap();
        match node {
            NodeType::Phi(_name) => {
                connect_phi_node_sel_id(ir, id, nm);
            }
            _ => {
                continue;
            }
        }
    }
}

fn is_reference(expr: &Expr) -> bool {
    match expr {
        Expr::Reference(_) => true,
        _ => false,
    }
}
fn is_reference_or_const(expr: &Expr) -> bool {
    match expr {
        Expr::Reference(_) => true,
        Expr::UIntInit(_, _) |
            Expr::SIntInit(_, _) => true,
        _ => false,
    }
}

fn is_reference_or_const_or_primop1expr(expr: &Expr) -> bool {
    match expr {
        Expr::Reference(_) => true,
        Expr::UIntInit(_, _) => true,
        Expr::SIntInit(_, _) => true,
        Expr::PrimOp1Expr(_, iexpr) => {
            match iexpr.as_ref() {
                Expr::UIntInit(_, _) => true,
                Expr::SIntInit(_, _) => true,
                _ => false,
            }
        },
        _ => false,
    }
}

pub fn check_ast_assumption(ast: &Circuit) {
    for cm in ast.modules.iter() {
        match cm.as_ref() {
            CircuitModule::Module(m) => {
                check_stmt_assumption(&m.stmts);
            }
            _ => { }
        }
    }
}

/// While converting the AST into graph form, we want to make sure
/// certain assumptions about the AST hold to make the conversion pass
/// scoped
fn check_stmt_assumption(stmts: &Stmts) {
    for stmt in stmts {
        match stmt.as_ref() {
            Stmt::When(cond, _info, when_stmts, else_stmts_opt) => {
                check_stmt_assumption(when_stmts);
                if let Some(else_stmts) = else_stmts_opt {
                    check_stmt_assumption(else_stmts);
                }
                assert!(is_reference_or_const(cond), "When condition should be a Reference {:?}", cond);
            }
            Stmt::Wire(_name, _tpe, _info) => {
            }
            Stmt::Reg(_name, _tpe, _clk, _info) => {
            }
            Stmt::RegReset(_name, _tpe, _clk, _rst, _init, _info) => {
            }
            Stmt::ChirrtlMemory(_mem) => {
                continue;
            }
            Stmt::ChirrtlMemoryPort(_mport) => {
                continue;
            }
            Stmt::Inst(_inst_name, _mod_name, _info) => {
                continue;
            }
            Stmt::Node(_out_name, expr, _info) => {
                match expr {
                    Expr::Mux(cond, true_expr, false_expr) => {
                        assert!(is_reference_or_const_or_primop1expr(cond), "Mux cond {:?} is not a Reference", cond);
                        assert!(is_reference_or_const_or_primop1expr(true_expr), "Mux true_expr {:?} is not a Reference", true_expr);
                        assert!(is_reference_or_const_or_primop1expr(false_expr), "Mux false_expr {:?} is not a Reference", false_expr);
                    }
                    Expr::PrimOp2Expr(op, a, b) => {
                        assert!(is_reference_or_const_or_primop1expr(a), "{:?} a {:?} is not a Reference", op, a);
                        assert!(is_reference_or_const_or_primop1expr(b), "{:?} b {:?} is not a Reference", op, b);
                    }
                    Expr::PrimOp1Expr(op, a) => {
                        assert!(is_reference_or_const_or_primop1expr(a), "{:?} a {:?} is not a Reference", op, a);
                    }
                    Expr::PrimOp1Expr1Int(op, a, _x) => {
                        assert!(is_reference_or_const_or_primop1expr(a), "{:?} a {:?} is not a Reference", op, a);
                    }
                    Expr::PrimOp1Expr2Int(op, a, _x, _y) => {
                        assert!(is_reference(a), "{:?} a {:?} is not a Reference", op, a);
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
            Stmt::Connect(lhs, rhs, _info) => {
                assert!(is_reference(lhs), "Connect lhs {:?} is not a Reference", lhs);
                assert!(is_reference_or_const_or_primop1expr(rhs), "Connect rhs {:?} is not a Reference", rhs);
            }
            Stmt::Invalidate(_expr, _info) => {
                continue;
            }
            Stmt::Skip(_) => {
                continue;
            }
            Stmt::Printf(_clk, _posedge, _msg, _exprs_opt, _info) => {
                continue;
            }
            Stmt::Assert(_clk, _pred, _cond, _msg, _info) => {
                continue;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{common::graphviz::GraphViz, parser::parse_circuit};

    #[test]
    fn gcd() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./examples/GCD.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let irs = from_circuit(&circuit);
        for ir in irs {
            ir.export_graphviz("./test-outputs/GCD.dot.pdf", None, true)?;
        }
        Ok(())
    }

    #[test]
    fn nestedwhen() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./examples/NestedWhen.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let irs = from_circuit(&circuit);
        for ir in irs {
            ir.export_graphviz("./test-outputs/NestedWhen.dot.pdf", None, true)?;
        }
        Ok(())
    }

    #[test]
    fn nestedbundle() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./examples/NestedBundleModule.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let irs = from_circuit(&circuit);
        for ir in irs {
            ir.export_graphviz("./test-outputs/NestedBundleModule.dot.pdf", None, true)?;
        }
        Ok(())
    }

    #[test]
    fn rocket_assumption() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./test-inputs/chipyard.harness.TestHarness.RocketConfig.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");
        check_ast_assumption(&circuit);
        Ok(())
    }

    #[test]
    fn boom_assumption() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./test-inputs/chipyard.harness.TestHarness.LargeBoomV3Config.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");
        check_ast_assumption(&circuit);
        Ok(())
    }
}
