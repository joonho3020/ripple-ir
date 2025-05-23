use rusty_firrtl::*;
use indexmap::IndexMap;

pub fn is_reference(expr: &Expr) -> bool {
    match expr {
        Expr::Reference(_) => true,
        _ => false,
    }
}

pub fn is_reference_or_const(expr: &Expr) -> bool {
    match expr {
        Expr::Reference(_) => true,
        Expr::UIntInit(_, _) |
            Expr::SIntInit(_, _) => true,
        _ => false,
    }
}

pub fn is_reference_or_const_or_primop1expr(expr: &Expr) -> bool {
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

/// While converting the AST into graph form, we want to make sure
/// certain assumptions about the AST hold to make the conversion pass
/// scoped
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

pub fn check_stmt_assumption(stmts: &Stmts) {
    let mut reference_srcs: IndexMap<&Identifier, &Stmt> = IndexMap::new();

    for stmt in stmts {
        match stmt.as_ref() {
            Stmt::When(cond, _info, when_stmts, else_stmts_opt) => {
                check_stmt_assumption(when_stmts);
                if let Some(else_stmts) = else_stmts_opt {
                    check_stmt_assumption(else_stmts);
                }
                assert!(is_reference_or_const(cond), "When condition should be a Reference {:?}", cond);
            }
            Stmt::Wire(name, _tpe, _info) => {
                reference_srcs.insert(name, stmt.as_ref());
            }
            Stmt::Reg(name, _tpe, _clk, _info) => {
                reference_srcs.insert(name, stmt.as_ref());
            }
            Stmt::RegReset(name, _tpe, _clk, _rst, _init, _info) => {
                reference_srcs.insert(name, stmt.as_ref());
            }
            Stmt::ChirrtlMemory(_mem) => {
                continue;
            }
            Stmt::ChirrtlMemoryPort(mport) => {
                match mport {
                    ChirrtlMemoryPort::Write(port, _mem, addr, _clk, _info) |
                        ChirrtlMemoryPort::Read (port, _mem, addr, _clk, _info) |
                        ChirrtlMemoryPort::Infer(port, _mem, addr, _clk, _info) => {
                            reference_srcs.insert(port, stmt.as_ref());
                            assert!(is_reference_or_const_or_primop1expr(addr), "MemPort addr {:?} is not a Reference", addr);
                    }
                }
            }
            Stmt::Inst(_inst_name, _mod_name, _info) => {
                continue;
            }
            Stmt::Node(name, expr, _info) => {
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
                reference_srcs.insert(name, stmt.as_ref());
            }
            Stmt::Connect(lhs, rhs, _info) => {
                assert!(is_reference(lhs), "Connect lhs {:?} is not a Reference", lhs);
                assert!(is_reference_or_const_or_primop1expr(rhs), "Connect rhs {:?} is not a Reference", rhs);
                if let Expr::Reference(r) = lhs {
                    if let Some(decl_stmt) = reference_srcs.get(&r.root()) {
                        match *decl_stmt {
                            Stmt::Reg(..)                   |
                                Stmt::Wire(..)              |
                                Stmt::RegReset(..)          |
                                Stmt::Inst(..)              |
                                Stmt::ChirrtlMemoryPort(..) => {
                            }
                            _ => {
                                assert!(false, "Unexpected lhs src stmt {:?}", lhs);
                            }
                        }
                    }
                }
            }
            Stmt::Invalidate(_expr, _info) => {
                continue;
            }
            Stmt::Skip(_) => {
                continue;
            }
            Stmt::Printf(..) => {
                continue;
            }
            Stmt::Assert(..) => {
                continue;
            }
            Stmt::Stop(..) => {
                unimplemented!();
            }
            Stmt::Memory(..) => {
                unimplemented!();
            }
        }
    }
}
