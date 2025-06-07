use rusty_firrtl::*;
use indexmap::IndexSet;
use indexmap::IndexMap;
use serde_json::Value;
use crate::ir::fir::NameSpace;

/// Looks at the FIRRTL3 stmts and adds a node if there is a expr on the rhs that is not a
/// reference
/// - Assumes that the input FIRRTL expression is in lo form
pub fn firrtl3_split_exprs(circuit: &mut Circuit, annos: &mut Option<Annotations>) {
    for cm in circuit.modules.iter_mut() {
        match cm.as_mut() {
            CircuitModule::Module(module) => {
                firrtl3_split_exprs_module(module);
                firrtl3_replace_rename_nodes(module, annos);
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
        Expr::PrimOp1Expr(op, a) => {
            match op {
                PrimOp1Expr::AsUInt |
                    PrimOp1Expr::AsSInt |
                    PrimOp1Expr::AsClock |
                    PrimOp1Expr::AsAsyncReset |
                    PrimOp1Expr::Neg |
                    PrimOp1Expr::Not |
                    PrimOp1Expr::Orr |
                    PrimOp1Expr::Andr |
                    PrimOp1Expr::Xorr => {
                        let (a_split,  mut a_stmts) = split_nested_exprs_recursive(a, ns);
                        stmts.append(&mut a_stmts);

                        let split_expr = Expr::PrimOp1Expr(*op, Box::new(a_split));
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
        Expr::PrimOp1Expr1Int(op, a, x) => {
            let (a_split,  mut a_stmts) = split_nested_exprs_recursive(a, ns);
            stmts.append(&mut a_stmts);

            let split_expr = Expr::PrimOp1Expr1Int(*op, Box::new(a_split), x.clone());
            let split_expr_name = ns.next();
            let split_stmt = Stmt::Node(split_expr_name.clone(), split_expr, Info::default());
            stmts.push(Box::new(split_stmt));

            let ref_expr = Expr::Reference(Reference::Ref(split_expr_name.clone()));
            return (ref_expr, stmts);
        }
        Expr::PrimOp1Expr2Int(op, a, x, y) => {
            let (a_split,  mut a_stmts) = split_nested_exprs_recursive(a, ns);
            stmts.append(&mut a_stmts);

            let split_expr = Expr::PrimOp1Expr2Int(*op, Box::new(a_split), x.clone(), y.clone());
            let split_expr_name = ns.next();
            let split_stmt = Stmt::Node(split_expr_name.clone(), split_expr, Info::default());
            stmts.push(Box::new(split_stmt));

            let ref_expr = Expr::Reference(Reference::Ref(split_expr_name.clone()));
            return (ref_expr, stmts);
        }
        Expr::ValidIf(a, b) => {
            let (a_split,  mut a_stmts) = split_nested_exprs_recursive(a, ns);
            let (b_split,  mut b_stmts) = split_nested_exprs_recursive(b, ns);
            stmts.append(&mut a_stmts);
            stmts.append(&mut b_stmts);

            let split_expr = Expr::ValidIf(Box::new(a_split), Box::new(b_split));
            let split_expr_name = ns.next();
            let split_stmt = Stmt::Node(split_expr_name.clone(), split_expr, Info::default());
            stmts.push(Box::new(split_stmt));

            let ref_expr = Expr::Reference(Reference::Ref(split_expr_name.clone()));
            return (ref_expr, stmts);
        }
        Expr::Reference(_) |
            Expr::UIntInit(..) |
            Expr::SIntInit(..) |
            Expr::UIntNoInit(..) |
            Expr::SIntNoInit(..) => {
            return (expr.clone(), stmts);
        }
    }
}

fn replace_rename_node_reference(reference: &Reference, rename_map: &IndexMap<&Identifier, ReferenceOrConst>) -> Expr {
    match reference {
        Reference::Ref(name) => {
            if rename_map.contains_key(name) {
                let x = rename_map.get(name).unwrap();
                match x {
                    &ReferenceOrConst::Ref(r) => {
                        Expr::Reference(r.clone())  
                    }
                    &ReferenceOrConst::Const(c) => {
                        c.clone()
                    }
                }
            } else {
                Expr::Reference(reference.clone())
            }
        }
        Reference::RefDot(parent, name) => {
            if let Expr::Reference(parent_ref) = replace_rename_node_reference(parent.as_ref(), rename_map) {
                Expr::Reference(Reference::RefDot(
                        Box::new(parent_ref),
                        name.clone()))
            } else {
                unreachable!()
            }
        }
        Reference::RefIdxInt(parent, idx) => {
            if let Expr::Reference(parent_ref) = replace_rename_node_reference(parent.as_ref(), rename_map) {
                Expr::Reference(Reference::RefIdxInt(
                    Box::new(parent_ref),
                    idx.clone()))
            } else {
                unreachable!()
            }
        }
        Reference::RefIdxExpr(parent, idx_expr) => {
            if let Expr::Reference(parent_ref) = replace_rename_node_reference(parent.as_ref(), rename_map) {
                Expr::Reference(Reference::RefIdxExpr(
                    Box::new(parent_ref),
                    Box::new(replace_rename_node_expr(idx_expr.as_ref(), rename_map))))
            } else {
                unreachable!()
            }
        }
    }
}

fn replace_rename_node_expr(expr: &Expr, rename_map: &IndexMap<&Identifier, ReferenceOrConst>) -> Expr {
    match expr {
        Expr::Reference(reference) => {
            replace_rename_node_reference(reference, rename_map)
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

enum ReferenceOrConst<'a> {
    Ref(&'a Reference),
    Const(&'a Expr)
}

impl<'a> ReferenceOrConst<'a> {
    fn from_ref(r: &'a Reference) -> ReferenceOrConst<'a> {
        ReferenceOrConst::Ref(r)
    }
    fn from_const(c: &'a Expr) -> ReferenceOrConst<'a> {
        ReferenceOrConst::Const(c)
    }
}


fn replace_reference_in_annos(
    cur_module: &Identifier,
    rename_map: &IndexMap<&Identifier, ReferenceOrConst>,
    annos_opt: &mut Option<Annotations>,
) {
    if let Some(annos) = annos_opt {
        if let Some(annos_list) = annos.0.as_array_mut() {
            let mut new_list = vec![];
            for mut anno in annos_list.drain(..) {
                if let Some(map) = anno.as_object_mut() {
                    let keep = update_target_field(map, cur_module, rename_map);
                    if keep {
                        new_list.push(anno);
                    }
                } else {
                    new_list.push(anno);
                }
            }
            annos_list.extend(new_list);
        }
    }
}

fn update_target_field(
    map: &mut serde_json::Map<String, Value>,
    cur_module: &Identifier,
    rename_map: &IndexMap<&Identifier, ReferenceOrConst>,
) -> bool {
    if let Some(Value::String(value_str)) = map.get("target") {
        let parts: Vec<&str> = value_str.split(|c| c == '|' || c == '>').collect();
        if parts.len() == 3 &&
            rename_map.contains_key(&Identifier::Name(parts[2].to_string()))
        {
            let path = parse_path(parts[1]);
            if path.last().unwrap().module == cur_module.to_string() {
                match &rename_map[&Identifier::Name(parts[2].to_string())] {
                    ReferenceOrConst::Ref(renamed_ref) => {
                        let new_target = format!("{}|{}>{}", parts[0], parts[1], renamed_ref);
                        map.insert("target".to_string(), Value::String(new_target));
                        return true;
                    }
                    ReferenceOrConst::Const(_) => {
                        return false;
                    }
                }
            }
        }
    }

    true
}

#[derive(Debug)]
struct InstPath {
    module: String,
    _inst: Option<String>,
}

fn parse_path(path: &str) -> Vec<InstPath> {
    path.split('/')
        .map(|segment| {
            let parts: Vec<&str> = segment.split(':').collect();
            match parts.len() {
                2 => InstPath {
                    _inst: Some(parts[0].to_string()),
                    module: parts[1].to_string(),
                },
                1 => InstPath {
                    _inst: None,
                    module: parts[0].to_string(),
                },
                _ => panic!("Invalid segment format: {}", segment),
            }
        })
        .collect()
}



fn firrtl3_replace_rename_nodes(module: &mut Module, annos_opt: &mut Option<Annotations>) {
    let mut rename_node_names: IndexMap<&Identifier, ReferenceOrConst> = IndexMap::new();
    let mut stmts = vec![];
    for stmt in module.stmts.iter() {
        match stmt.as_ref() {
            Stmt::Connect(lhs, rhs, info) => {
                stmts.push(Box::new(
                    Stmt::Connect(
                        lhs.clone(),
                        replace_rename_node_expr(rhs, &rename_node_names),
                        info.clone())));
            }
            Stmt::Node(name, expr, info) => {
                match expr {
                    Expr::Reference(reference) => {
                        rename_node_names.insert(name, ReferenceOrConst::from_ref(reference));
                    }
                    Expr::UIntInit(..) |
                        Expr::SIntInit(..) => {
                            rename_node_names.insert(name, ReferenceOrConst::from_const(expr));
                    }
                    _ => {
                        stmts.push(Box::new(
                            Stmt::Node(
                                name.clone(),
                                replace_rename_node_expr(expr, &rename_node_names),
                                info.clone())
                            ));
                    }
                }
            }
            Stmt::Reg(name, tpe, clk, info) => {
                stmts.push(Box::new(
                    Stmt::Reg(name.clone(), tpe.clone(),
                    replace_rename_node_expr(clk, &rename_node_names),
                    info.clone())));


            }
            Stmt::RegReset(name, tpe, clk, rst, init, info) => {
                stmts.push(Box::new(
                    Stmt::RegReset(name.clone(), tpe.clone(),
                    replace_rename_node_expr(clk, &rename_node_names),
                    replace_rename_node_expr(rst, &rename_node_names),
                    replace_rename_node_expr(init, &rename_node_names),
                    info.clone())));
            }
            Stmt::Printf(name_opt, clk, trig, msg, args_opt, info) => {
                let new_args = if let Some(args) = args_opt {
                    let mut ret = vec![];
                    for arg in args {
                        ret.push(Box::new(replace_rename_node_expr(arg.as_ref(), &rename_node_names)));
                    }
                    Some(ret)
                } else {
                    None
                };

                stmts.push(Box::new(
                        Stmt::Printf(name_opt.clone(),
                        replace_rename_node_expr(clk, &rename_node_names),
                        replace_rename_node_expr(trig, &rename_node_names),
                        msg.clone(),
                        new_args,
                        info.clone())));
            }
            _ => {
                stmts.push(stmt.clone());
            }
        }
    }
    replace_reference_in_annos(&module.name, &rename_node_names, annos_opt);
    module.stmts = stmts;
}
