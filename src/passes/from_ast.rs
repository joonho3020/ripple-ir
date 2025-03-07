use crate::{ir::*, parser::ast::*};
use crate::parser::typetree::{Direction, TypeTree};
use petgraph::graph::NodeIndex;
use indexmap::IndexMap;

type RefMap = IndexMap<Reference, NodeIndex>;

#[derive(Default, Debug)]
struct AST2GraphConverter {
}

impl AST2GraphConverter {
    pub fn from_circuit(ast: &Circuit) -> Vec<RippleIR> {
        let mut ret = vec![];
        for module in ast.modules.iter() {
            ret.push(Self::from_circuit_module(module));
        }
        return ret;
    }

    fn from_circuit_module(module: &CircuitModule) -> RippleIR {
        let mut refmap: RefMap = RefMap::new();
        match module {
            CircuitModule::Module(m) => {
                Self::from_module(m, &mut refmap)
            }
            CircuitModule::ExtModule(e) => {
                Self::from_ext_module(e, &mut refmap)
            }
        }
    }

    fn from_module(module: &Module, refmap: &mut RefMap) -> RippleIR {
        let mut ret = RippleIR::new();
        Self::collect_ports(&mut ret, &module.ports, refmap);
        Self::collect_node_stmts(&mut ret, &module.stmts, refmap);
        Self::connect_nodes_stmts(&mut ret, &module.stmts, refmap);
        return ret;
    }

    fn from_ext_module(module: &ExtModule, refmap: &mut RefMap) -> RippleIR {
        let ret = RippleIR::new();
        return ret;
    }

    fn all_references(tpe: &Type, name: &Identifier, dir: Option<Direction>) -> Vec<Reference> {
        let typetree = tpe.construct_tree(name.clone(), dir.unwrap_or(Direction::Output));
        typetree.all_possible_references()
    }

    fn collect_ports(ir: &mut RippleIR, ports: &Ports, refmap: &mut RefMap) {
        for port in ports.iter() {
            match port.as_ref() {
                Port::Input(name, tpe, _info) => {
                    let id = ir.graph.add_node(NodeType::Input(name.clone(), tpe.clone()));
                    let all_refs = Self::all_references(tpe, name, Some(Direction::Input));
                    for reference in all_refs {
                        refmap.insert(reference, id);
                    }
                }
                Port::Output(name, tpe, _info) => {
                    let id = ir.graph.add_node(NodeType::Output(name.clone(), tpe.clone()));
                    let all_refs = Self::all_references(tpe, name, Some(Direction::Output));
                    for reference in all_refs {
                        refmap.insert(reference, id);
                    }
                }
            }
        }
    }

    fn collect_node_exprs(ir: &mut RippleIR, expr: &Expr, refmap: &mut RefMap) {
        match expr {
            Expr::UIntInit(w, init) => {
                ir.graph.add_node(NodeType::UIntLiteral(*w, init.clone()));
            }
            Expr::SIntInit(w, init) => {
                ir.graph.add_node(NodeType::SIntLiteral(*w, init.clone()));
            }
            Expr::Mux(cond, true_expr, false_expr) => {
                ir.graph.add_node(NodeType::Mux);
                Self::collect_node_exprs(ir, cond, refmap);
                Self::collect_node_exprs(ir, true_expr, refmap);
                Self::collect_node_exprs(ir, false_expr, refmap);
            }
            Expr::PrimOp2Expr(op, a, b) => {
                ir.graph.add_node(NodeType::PrimOp2Expr(*op));
                Self::collect_node_exprs(ir, a, refmap);
                Self::collect_node_exprs(ir, b, refmap);
            }
            Expr::PrimOp1Expr(op, a) => {
                ir.graph.add_node(NodeType::PrimOp1Expr(*op));
                Self::collect_node_exprs(ir, a, refmap);
            }
            Expr::PrimOp1Expr1Int(op, a, x) => {
                ir.graph.add_node(NodeType::PrimOp1Expr1Int(*op, x.clone()));
                Self::collect_node_exprs(ir, a, refmap);
            }
            Expr::PrimOp1Expr2Int(op, a, x, y) => {
                ir.graph.add_node(NodeType::PrimOp1Expr2Int(*op, x.clone(), y.clone()));
                Self::collect_node_exprs(ir, a, refmap);
            }
            Expr::Reference(_) => {
                // TODO: RefExpr?
                return;
            }
            Expr::UIntNoInit(_) |
                Expr::SIntNoInit(_) |
                Expr::ValidIf(_, _) => {
                panic!("collect_node_exprs doesn't handle expression {:?}", expr);
            }
        }
    }

    fn collect_node_stmts(ir: &mut RippleIR, stmts: &Stmts, refmap: &mut RefMap) {
        for stmt in stmts {
            Self::collect_node_stmt(ir, stmt.as_ref(), refmap);
        }
    }

    fn collect_node_stmt(ir: &mut RippleIR, stmt: &Stmt, refmap: &mut RefMap) {
        match stmt {
            Stmt::When(cond, _info, when_stmts, else_stmts_opt) => {
                Self::collect_node_exprs(ir, cond, refmap);
                Self::collect_node_stmts(ir, when_stmts, refmap);
                if let Some(else_stmts) = else_stmts_opt {
                    Self::collect_node_stmts(ir, else_stmts, refmap);
                }
            }
            Stmt::Wire(name, tpe, _info) => {
                let id = ir.graph.add_node(NodeType::Wire(name.clone(), tpe.clone()));
                let all_refs = Self::all_references(tpe, name, None);
                for reference in all_refs {
                    refmap.insert(reference, id);
                }
            }
            Stmt::Reg(name, tpe, clk, _info) => {
                let id = ir.graph.add_node(NodeType::Reg(name.clone(), tpe.clone(), clk.clone()));
                let all_refs = Self::all_references(tpe, name, None);
                for reference in all_refs {
                    refmap.insert(reference, id);
                }
            }
            Stmt::RegReset(name, tpe, clk, rst, init, _info) => {
                let id = ir.graph.add_node(NodeType::RegReset(
                        name.clone(), tpe.clone(), clk.clone(), rst.clone(), init.clone()));
                let all_refs = Self::all_references(tpe, name, None);
                for reference in all_refs {
                    refmap.insert(reference, id);
                }
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
                match expr {
                    Expr::Mux(_, _, _) => {
                        let id = ir.graph.add_node(NodeType::Mux);
                        refmap.insert(Reference::Ref(out_name.clone()), id);
                    }
                    Expr::PrimOp2Expr(op, _, _) => {
                        let id = ir.graph.add_node(NodeType::PrimOp2Expr(*op));
                        refmap.insert(Reference::Ref(out_name.clone()), id);
                    }
                    Expr::PrimOp1Expr(op, _) => {
                        let id = ir.graph.add_node(NodeType::PrimOp1Expr(*op));
                        refmap.insert(Reference::Ref(out_name.clone()), id);
                    }
                    Expr::PrimOp1Expr1Int(op, _, x) => {
                        let id = ir.graph.add_node(NodeType::PrimOp1Expr1Int(*op, x.clone()));
                        refmap.insert(Reference::Ref(out_name.clone()), id);
                    }
                    Expr::PrimOp1Expr2Int(op, _, x, y) => {
                        let id = ir.graph.add_node(NodeType::PrimOp1Expr2Int(*op, x.clone(), y.clone()));
                        refmap.insert(Reference::Ref(out_name.clone()), id);
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
            Stmt::Connect(..) => {
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

    fn connect_nodes_stmts(ir: &mut RippleIR, stmts: &Stmts, refmap: &mut RefMap) {
        for stmt in stmts {
            Self::connect_nodes_stmt(ir, stmt.as_ref(), refmap);
        }
    }

    fn connect_dst_to_expr(ir: &mut RippleIR, dst_id: NodeIndex, rhs: &Expr, refmap: &RefMap) {
        match rhs {
            Expr::Reference(r) => {
                let src_id = refmap.get(r).expect(&format!("Connect rhs not found in refmap {:?}", r));
                ir.graph.add_edge(*src_id, dst_id, EdgeType::Default);
            }
            Expr::UIntInit(w, init) => {
                let src_id = ir.graph.add_node(NodeType::UIntLiteral(*w, init.clone()));
                ir.graph.add_edge(src_id, dst_id, EdgeType::Default);
            }
            Expr::SIntInit(w, init) => {
                let src_id = ir.graph.add_node(NodeType::SIntLiteral(*w, init.clone()));
                ir.graph.add_edge(src_id, dst_id, EdgeType::Default);
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
                ir.graph.add_edge(op_id, dst_id, EdgeType::Default);
                ir.graph.add_edge(src_id, op_id, EdgeType::Default);
            }
            _ => {
                assert!(false, "Connect rhs {:?} unhandled type", rhs);
            }
        }
    }

    fn connect_nodes_stmt(ir: &mut RippleIR, stmt: &Stmt, refmap: &mut RefMap) {
        match stmt {
            Stmt::When(cond, _info, when_stmts, else_stmts_opt) => {
                Self::connect_nodes_stmts(ir, when_stmts, refmap);
                if let Some(else_stmts) = else_stmts_opt {
                    Self::connect_nodes_stmts(ir, else_stmts, refmap);
                }
                assert!(Self::is_reference_or_const(cond), "When condition should be a Reference {:?}", cond);
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
                let dst_id = refmap.get(&Reference::Ref(out_name.clone()))
                    .expect(&format!("Node lhs not found in refmap {:?}", stmt));

                match expr {
                    Expr::Mux(cond, true_expr, false_expr) => {
                        Self::connect_dst_to_expr(ir, *dst_id, cond, refmap);
                        Self::connect_dst_to_expr(ir, *dst_id, &true_expr, refmap);
                        Self::connect_dst_to_expr(ir, *dst_id, &false_expr, refmap);
                    }
                    Expr::PrimOp2Expr(_, a, b) => {
                        Self::connect_dst_to_expr(ir, *dst_id, &a, refmap);
                        Self::connect_dst_to_expr(ir, *dst_id, &b, refmap);
                    }
                    Expr::PrimOp1Expr(_, a) => {
                        Self::connect_dst_to_expr(ir, *dst_id, &a, refmap);
                    }
                    Expr::PrimOp1Expr1Int(_, a, _) => {
                        Self::connect_dst_to_expr(ir, *dst_id, &a, refmap);
                    }
                    Expr::PrimOp1Expr2Int(_, a, _, _) => {
                        Self::connect_dst_to_expr(ir, *dst_id, &a, refmap);
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
            Stmt::Connect(lhs, rhs, _info) => {
                let dst_id = match lhs {
                    Expr::Reference(r) => {
                        refmap.get(r).expect(&format!("Connect lhs not found in refmap {:?}", lhs))
                    }
                    _ => {
                        panic!("Connect lhs {:?} unhandled type", lhs);
                    }
                };
                Self::connect_dst_to_expr(ir, *dst_id, rhs, refmap);
            }
            Stmt::Printf(_clk, _posedge, _msg, _exprs_opt, _info) => {
            }
            Stmt::Assert(_clk, _pred, _cond, _msg, _info) => {
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
                    Self::check_stmt_assumption(&m.stmts);
                }
                _ => { }
            }
        }
    }

    fn check_stmt_assumption(stmts: &Stmts) {
        for stmt in stmts {
            match stmt.as_ref() {
                Stmt::When(cond, _info, when_stmts, else_stmts_opt) => {
                    Self::check_stmt_assumption(when_stmts);
                    if let Some(else_stmts) = else_stmts_opt {
                        Self::check_stmt_assumption(else_stmts);
                    }
                    assert!(Self::is_reference_or_const(cond), "When condition should be a Reference {:?}", cond);
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
                            assert!(Self::is_reference_or_const_or_primop1expr(cond), "Mux cond {:?} is not a Reference", cond);
                            assert!(Self::is_reference_or_const_or_primop1expr(true_expr), "Mux true_expr {:?} is not a Reference", true_expr);
                            assert!(Self::is_reference_or_const_or_primop1expr(false_expr), "Mux false_expr {:?} is not a Reference", false_expr);
                        }
                        Expr::PrimOp2Expr(op, a, b) => {
                            assert!(Self::is_reference_or_const_or_primop1expr(a), "{:?} a {:?} is not a Reference", op, a);
                            assert!(Self::is_reference_or_const_or_primop1expr(b), "{:?} b {:?} is not a Reference", op, b);
                        }
                        Expr::PrimOp1Expr(op, a) => {
                            assert!(Self::is_reference_or_const_or_primop1expr(a), "{:?} a {:?} is not a Reference", op, a);
                        }
                        Expr::PrimOp1Expr1Int(op, a, _x) => {
                            assert!(Self::is_reference_or_const_or_primop1expr(a), "{:?} a {:?} is not a Reference", op, a);
                        }
                        Expr::PrimOp1Expr2Int(op, a, _x, _y) => {
                            assert!(Self::is_reference(a), "{:?} a {:?} is not a Reference", op, a);
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
                    assert!(Self::is_reference(lhs), "Connect lhs {:?} is not a Reference", lhs);
                    assert!(Self::is_reference_or_const_or_primop1expr(rhs), "Connect rhs {:?} is not a Reference", rhs);
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
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{common::graphviz::GraphViz, parser::parse_circuit};

    #[test]
    fn gcd() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./examples/GCD.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let irs = AST2GraphConverter::from_circuit(&circuit);
        for ir in irs {
            let dot = ir.export_graphviz("./test-outputs/GCD.dot.pdf", None)?;
            println!("{:#?}", dot);
        }
        Ok(())
    }

    #[test]
    fn rocket_assumption() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./test-inputs/chipyard.harness.TestHarness.RocketConfig.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");
        AST2GraphConverter::check_ast_assumption(&circuit);
        Ok(())
    }

    #[test]
    fn boom_assumption() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./test-inputs/chipyard.harness.TestHarness.LargeBoomV3Config.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");
        AST2GraphConverter::check_ast_assumption(&circuit);
        Ok(())
    }
}
