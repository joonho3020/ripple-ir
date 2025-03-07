use petgraph::data::Build;

use crate::{ir::*, parser::ast::*};
use crate::parser::typetree::{Direction, TypeTree};

#[derive(Default, Debug)]
struct AST2GraphConverter {
}

impl AST2GraphConverter {
    pub fn from_circuit(&mut self, ast: &Circuit) -> Vec<RippleIR> {
        let mut ret = vec![];
        for module in ast.modules.iter() {
            ret.push(self.from_circuit_module(module));
        }
        return ret;
    }

    fn from_circuit_module(&mut self, module: &CircuitModule) -> RippleIR {
        match module {
            CircuitModule::Module(m) => {
                self.from_module(m)
            }
            CircuitModule::ExtModule(e) => {
                self.from_ext_module(e)
            }
        }
    }

    fn from_module(&mut self, module: &Module) -> RippleIR {
        let mut ret = RippleIR::new();
        self.collect_ports(&mut ret, &module.ports);
        self.collect_node_stmts(&mut ret, &module.stmts);
        println!("from module done");
        return ret;
    }

    fn from_ext_module(&mut self, module: &ExtModule) -> RippleIR {
        let ret = RippleIR::new();
        return ret;
    }

    fn collect_ports(&mut self, ir: &mut RippleIR, ports: &Ports) {
        println!("collect_ports {:?}", ports);
        for port in ports.iter() {
            let type_tree = match port.as_ref() {
                Port::Input(name, tpe, _info) => {
                    tpe.construct_tree(name.clone(), Direction::Input)
                }
                Port::Output(name, tpe, _info) => {
                    tpe.construct_tree(name.clone(), Direction::Output)
                }
            };
            let leaves = type_tree.collect_leaf_nodes();
            for leaf in leaves {
                let node = type_tree.get(leaf);
                let name = node.name.clone();
                let tpe = node.tpe.clone().unwrap();
                match node.dir {
                    Direction::Input => {
                        ir.graph.add_node(NodeType::Input(name, tpe));
                    }
                    Direction::Output => {
                        ir.graph.add_node(NodeType::Output(name, tpe));
                    }
                }
            }
        }
    }

    fn collect_node_exprs(&mut self, ir: &mut RippleIR, expr: &Expr) {
        println!("collect_expr_stmts {:?}", expr);
        match expr {
            Expr::UIntInit(w, init) => {
                ir.graph.add_node(NodeType::UIntLiteral(*w, init.clone()));
            }
            Expr::SIntInit(w, init) => {
                ir.graph.add_node(NodeType::SIntLiteral(*w, init.clone()));
            }
            Expr::Mux(cond, true_expr, false_expr) => {
                ir.graph.add_node(NodeType::Mux);
                self.collect_node_exprs(ir, cond);
                self.collect_node_exprs(ir, true_expr);
                self.collect_node_exprs(ir, false_expr);
            }
            Expr::PrimOp2Expr(op, a, b) => {
                ir.graph.add_node(NodeType::PrimOp2Expr(op.clone()));
                self.collect_node_exprs(ir, a);
                self.collect_node_exprs(ir, b);
            }
            Expr::PrimOp1Expr(op, a) => {
                ir.graph.add_node(NodeType::PrimOp1Expr(op.clone()));
                self.collect_node_exprs(ir, a);
            }
            Expr::PrimOp1Expr1Int(op, a, x) => {
                ir.graph.add_node(NodeType::PrimOp1Expr1Int(op.clone(), x.clone()));
                self.collect_node_exprs(ir, a);
            }
            Expr::PrimOp1Expr2Int(op, a, x, y) => {
                ir.graph.add_node(NodeType::PrimOp1Expr2Int(op.clone(), x.clone(), y.clone()));
                self.collect_node_exprs(ir, a);
            }
            Expr::Reference(_) => {
                return;
            }
            Expr::UIntNoInit(_) |
                Expr::SIntNoInit(_) |
                Expr::ValidIf(_, _) => {
                panic!("collect_node_exprs doesn't handle expression {:?}", expr);
            }
        }
    }

    fn collect_node_stmts(&mut self, ir: &mut RippleIR, stmts: &Stmts) {
        for stmt in stmts {
            println!("collect_node_stmts {:?}", stmt);
            match stmt.as_ref() {
                Stmt::When(cond, _info, when_stmts, else_stmts_opt) => {
                    self.collect_node_exprs(ir, cond);
                    self.collect_node_stmts(ir, when_stmts);
                    if let Some(else_stmts) = else_stmts_opt {
                        self.collect_node_stmts(ir, else_stmts);
                    }
                }
                Stmt::Reg(name, tpe, clk, _info) => {
                    let type_tree = tpe.construct_tree(name.clone(), Direction::Output);
                    let leaves = type_tree.collect_leaf_nodes();
                    for leaf in leaves {
                        let node = type_tree.get(leaf);
                        let name = node.name.clone();
                        let tpe = node.tpe.clone().unwrap();
                        ir.graph.add_node(NodeType::Reg(name, tpe, clk.clone()));
                    }
                }
                Stmt::RegReset(name, tpe, clk, rst, init, _info) => {
                    let type_tree = tpe.construct_tree(name.clone(), Direction::Output);
                    let leaves = type_tree.collect_leaf_nodes();
                    for leaf in leaves {
                        let node = type_tree.get(leaf);
                        let name = node.name.clone();
                        let tpe = node.tpe.clone().unwrap();
                        ir.graph.add_node(NodeType::RegReset(
                                name, tpe, clk.clone(), rst.clone(), init.clone()));
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
                Stmt::Node(_out_name, expr, _info) => {
                    self.collect_node_exprs(ir, expr);
                }
                Stmt::Connect(_lhs, _rhs, _info) => {
                    continue;
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
                _ => {
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

        let mut ast2graph = AST2GraphConverter::default();
        let irs = ast2graph.from_circuit(&circuit);
        for ir in irs {
            println!("ir {:?}", ir);
            ir.export_graphviz("./test-outputs/GCD.dot.pdf", None)?;
        }
        Ok(())
    }
}
