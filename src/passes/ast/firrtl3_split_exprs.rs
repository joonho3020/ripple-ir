use rusty_firrtl::*;
use indexmap::IndexSet;
use indexmap::IndexMap;
use crate::ir::fir::NameSpace;
use super::check_ast_assumption::is_reference_or_const_or_primop1expr;

/// Looks at the FIRRTL3 stmts and adds a node if there is a expr on the rhs that is not a
/// reference
/// - Assumes that the input FIRRTL expression is in lo form
pub fn firrtl3_split_exprs(circuit: &mut Circuit) {
    for cm in circuit.modules.iter_mut() {
        match cm.as_mut() {
            CircuitModule::Module(module) => {
                firrtl3_split_exprs_module(module);
                firrtl3_replace_rename_nodes(module);
            }
            CircuitModule::ExtModule(..) => {
                continue;
            }
        }
    }
}

fn firrtl3_split_exprs_module(module: &mut Module) {
    // Collecte used names on the LHS of a stmt
    let mut used: IndexSet<Identifier> = IndexSet::new();
    for stmt in module.stmts.iter() {
        match stmt.as_ref() {
            Stmt::Wire(name, ..)     |
            Stmt::Reg(name, ..)      |
            Stmt::RegReset(name, ..) |
            Stmt::Inst(name, ..)     |
            Stmt::Node(name, ..)     |
            Stmt::Memory(name, ..) => {
                used.insert(name.clone());
            }
            Stmt::Connect(lhs, ..) => {
                if let Expr::Reference(lhs_ref) = lhs {
                    used.insert(lhs_ref.root());
                }
            }
            _ => {
                continue;
            }
        }
    }

    // Create namespace
    let mut ns = NameSpace::from_used_set(used);

    let mut stmts = vec![];
    for stmt in module.stmts.iter() {
        match stmt.as_ref() {
            Stmt::Connect(lhs, rhs, info) => {
                match rhs {
                    Expr::Reference(..) |
                        Expr::UIntInit(..) |
                        Expr::SIntInit(..) => {
                        stmts.push(stmt.clone());
                    }
                    _ => {
                        // If a stmt has a connection where the RHS is not a reference
                        // (e.g., `maybe_full <= mux(reset, UInt<1>("h0"), _GEN_14)`)
                        // create a separate node stmt, and assign the node to the connection
                        let (split_expr, mut split_stmts) = split_nested_exprs(rhs, &mut ns);
                        stmts.append(&mut split_stmts);

                        let node_name = ns.next();
                        let node_stmt = Stmt::Node(node_name.clone(), split_expr.clone(), info.clone());
                        stmts.push(Box::new(node_stmt));

                        let node_ref = Expr::Reference(Reference::Ref(node_name));
                        let conn_stmt = Stmt::Connect(lhs.clone(), node_ref, info.clone());
                        stmts.push(Box::new(conn_stmt));
                    }
                }
            }
            Stmt::Node(name, expr, info) => {
                let (split_expr, mut split_stmts) = split_nested_exprs(expr, &mut ns);
                stmts.append(&mut split_stmts);

                let node_stmt = Stmt::Node(name.clone(), split_expr, info.clone());
                stmts.push(Box::new(node_stmt));
            }
            _ => {
                stmts.push(stmt.clone());
            }
        }
    }
    module.stmts = stmts;
}

fn split_nested_exprs(expr: &Expr, ns: &mut NameSpace) -> (Expr, Vec<Box<Stmt>>) {
    let mut stmts = vec![];
    match expr {
        Expr::Mux(cond, true_expr, false_expr) => {
            let (cond_split,  mut cond_stmts) = split_nested_exprs_recursive(cond, ns);
            let (true_split,  mut true_stmts) = split_nested_exprs_recursive(true_expr, ns);
            let (false_split, mut  false_stmts) = split_nested_exprs_recursive(false_expr, ns);

            stmts.append(&mut cond_stmts);
            stmts.append(&mut true_stmts);
            stmts.append(&mut false_stmts);

            let split_expr = Expr::Mux(
                Box::new(cond_split), Box::new(true_split), Box::new(false_split));
            return (split_expr, stmts);

        }
        Expr::PrimOp2Expr(op, a, b) => {
            let (a_split,  mut a_stmts) = split_nested_exprs_recursive(a, ns);
            let (b_split,  mut b_stmts) = split_nested_exprs_recursive(b, ns);

            stmts.append(&mut a_stmts);
            stmts.append(&mut b_stmts);

            let split_expr = Expr::PrimOp2Expr(op.clone(), Box::new(a_split), Box::new(b_split));
            return (split_expr, stmts);
        }
        _ => {
        }
    }
    (expr.clone(), stmts)
}

fn split_nested_exprs_recursive(expr: &Expr, ns: &mut NameSpace) -> (Expr, Vec<Box<Stmt>>) {
    let mut stmts = vec![];
    if is_reference_or_const_or_primop1expr(&expr) {
        return (expr.clone(), stmts);
    } else {
        match expr {
            Expr::Mux(cond, true_expr, false_expr) => {
                let (cond_split,  mut cond_stmts) = split_nested_exprs_recursive(cond, ns);
                let (true_split,  mut true_stmts) = split_nested_exprs_recursive(true_expr, ns);
                let (false_split, mut  false_stmts) = split_nested_exprs_recursive(false_expr, ns);
                stmts.append(&mut cond_stmts);
                stmts.append(&mut true_stmts);
                stmts.append(&mut false_stmts);

                let split_expr = Expr::Mux(Box::new(cond_split), Box::new(true_split), Box::new(false_split));
                let split_expr_name = ns.next();
                let split_stmt = Stmt::Node(split_expr_name.clone(), split_expr, Info::default());
                stmts.push(Box::new(split_stmt));

                let ref_expr = Expr::Reference(Reference::Ref(split_expr_name.clone()));
                return (ref_expr, stmts);
            }
            Expr::PrimOp2Expr(op, a, b) => {
                let (a_split,  mut a_stmts) = split_nested_exprs_recursive(a, ns);
                let (b_split,  mut b_stmts) = split_nested_exprs_recursive(b, ns);
                stmts.append(&mut a_stmts);
                stmts.append(&mut b_stmts);

                let split_expr = Expr::PrimOp2Expr(op.clone(), Box::new(a_split), Box::new(b_split));
                let split_expr_name = ns.next();
                let split_stmt = Stmt::Node(split_expr_name.clone(), split_expr, Info::default());
                stmts.push(Box::new(split_stmt));

                let ref_expr = Expr::Reference(Reference::Ref(split_expr_name.clone()));

                return (ref_expr, stmts);
            }
            _ => {
                return (expr.clone(), stmts);
            }
        }
    }
}

fn replace_rename_node_reference(reference: &Reference, rename_map: &IndexMap<&Identifier, &Reference>) -> Reference {
    match reference {
        Reference::Ref(name) => {
            if rename_map.contains_key(name) {
                let x = *rename_map.get(name).unwrap();
                x.clone()
            } else {
                reference.clone()
            }
        }
        Reference::RefDot(parent, name) => {
            Reference::RefDot(
                Box::new(replace_rename_node_reference(parent.as_ref(), rename_map)),
                name.clone())
        }
        Reference::RefIdxInt(parent, idx) => {
            Reference::RefIdxInt(
                Box::new(replace_rename_node_reference(parent.as_ref(), rename_map)),
                idx.clone())
        }
        Reference::RefIdxExpr(parent, idx_expr) => {
            Reference::RefIdxExpr(
                Box::new(replace_rename_node_reference(parent.as_ref(), rename_map)),
                Box::new(replace_rename_node_expr(idx_expr.as_ref(), rename_map)))
        }
    }
}

fn replace_rename_node_expr(expr: &Expr, rename_map: &IndexMap<&Identifier, &Reference>) -> Expr {
    match expr {
        Expr::Reference(reference) => {
            let replaced_ref = replace_rename_node_reference(reference, rename_map);
            Expr::Reference(replaced_ref)
        }
        Expr::Mux(cond, true_expr, false_expr) => {
            Expr::Mux(
                Box::new(replace_rename_node_expr(cond.as_ref(),       rename_map)),
                Box::new(replace_rename_node_expr(true_expr.as_ref(),  rename_map)),
                Box::new(replace_rename_node_expr(false_expr.as_ref(), rename_map)))
        }
        Expr::ValidIf(a, b) => {
            Expr::ValidIf(
                Box::new(replace_rename_node_expr(a.as_ref(),  rename_map)),
                Box::new(replace_rename_node_expr(b.as_ref(),  rename_map)))
        }
        Expr::PrimOp2Expr(op, a, b) => {
            Expr::PrimOp2Expr(
                *op,
                Box::new(replace_rename_node_expr(a.as_ref(),  rename_map)),
                Box::new(replace_rename_node_expr(b.as_ref(),  rename_map)))
        }
        Expr::PrimOp1Expr(op, a) => {
            Expr::PrimOp1Expr(
                *op,
                Box::new(replace_rename_node_expr(a.as_ref(),  rename_map)))
        }
        Expr::PrimOp1Expr1Int(op, a, x) => {
            Expr::PrimOp1Expr1Int(
                *op,
                Box::new(replace_rename_node_expr(a.as_ref(),  rename_map)),
                x.clone())
        }
        Expr::PrimOp1Expr2Int(op, a, x, y) => {
            Expr::PrimOp1Expr2Int(
                *op,
                Box::new(replace_rename_node_expr(a.as_ref(),  rename_map)),
                x.clone(),
                y.clone())
        }
        _ => {
            expr.clone()
        }
    }
}

fn firrtl3_replace_rename_nodes(module: &mut Module) {
    let mut rename_node_names: IndexMap<&Identifier, &Reference> = IndexMap::new();
    let mut stmts = vec![];
    for stmt in module.stmts.iter() {
        match stmt.as_ref() {
            Stmt::Connect(lhs, rhs, info) => {
                stmts.push(
                    Box::new(Stmt::Connect(
                        lhs.clone(),
                        replace_rename_node_expr(rhs, &rename_node_names),
                        info.clone())));
            }
            Stmt::Node(name, expr, info) => {
                match expr {
                    Expr::Reference(reference) => {
                        rename_node_names.insert(name, reference);
                    }
                    _ => {
                        stmts.push(
                            Box::new(Stmt::Node(
                                name.clone(),
                                replace_rename_node_expr(expr, &rename_node_names),
                                info.clone())
                            ));
                    }
                }
            }
            _ => {
                stmts.push(stmt.clone());
            }
        }
    }
    module.stmts = stmts;
}
