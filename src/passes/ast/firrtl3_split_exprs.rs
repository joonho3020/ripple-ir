use rusty_firrtl::*;
use indexmap::IndexSet;
use crate::ir::fir::NameSpace;

/// Looks at the FIRRTL3 stmts and adds a node if there is a expr on the rhs that is not a
/// reference
/// - Assumes that the input FIRRTL expression is in lo form
pub fn firrtl3_split_exprs(circuit: &mut Circuit) {
    for cm in circuit.modules.iter_mut() {
        match cm.as_mut() {
            CircuitModule::Module(module) => {
                firrtl3_split_exprs_module(module);
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
                        let node_name = ns.next();
                        let node_stmt = Stmt::Node(node_name.clone(), rhs.clone(), info.clone());
                        stmts.push(Box::new(node_stmt));

                        let node_ref = Expr::Reference(Reference::Ref(node_name));
                        let conn_stmt = Stmt::Connect(lhs.clone(), node_ref, info.clone());
                        stmts.push(Box::new(conn_stmt));
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
