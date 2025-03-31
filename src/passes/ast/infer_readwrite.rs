use chirrtl_parser::ast::*;
use indexmap::IndexMap;

pub type Connects = IndexMap<Identifier, Expr>;

pub type ReadWriteMemMap = IndexMap<Identifier, Identifier>;

fn get_connects_modules(m: &Module) -> Connects {
    let mut connect = Connects::new();
    get_connects_stmts(&m.stmts, &mut connect);
    connect
}

fn get_connects_stmts(stmts: &Stmts, connects: &mut Connects) {
    for stmt in stmts {
        match stmt.as_ref() {
            Stmt::Connect(lhs, rhs, _) => {
                if let Expr::Reference(r) = lhs {
                    connects.insert(r.root().clone(), rhs.clone());
                } else {
                    panic!("LHS of stmt should be a reference type");
                }
            }
            Stmt::When(_, _, when_stmts, else_stmts_opt) => {
                get_connects_stmts(when_stmts, connects);
                if let Some(else_stmts) = else_stmts_opt {
                    get_connects_stmts(else_stmts, connects);
                }
            }
            _ => { }
        }
    }
}

pub fn infer_read_write(circuit: &Circuit) {
    for module in circuit.modules.iter() {
        infer_read_write_module(module.as_ref());
    }
}

fn infer_read_write_module(cm: &CircuitModule) {
    match cm {
        CircuitModule::Module(m) => {
            let connects = get_connects_modules(m);
        }
        _ => { }
    }
}

// fn infer_read_write_stmts(connects: &Connects, stmts: &Stmts, readwrite: &mut ReadWriteMemMap) {
// for stmt in stmts {
// ratch stmt.as_ref() {
// Stmt::ChirrtlMemory(x)
// }
// }
// }
