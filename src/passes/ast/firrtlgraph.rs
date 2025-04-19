use chirrtl_parser::ast::*;
use petgraph::{graph::{Graph, NodeIndex}, visit::EdgeRef};
use crate::passes::ast::print::Printer;

/// A node in the FIRRTL AST for GumTree comparison
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FirrtlNode {
    Circuit(Version, Identifier, Annotations),
    Module(Identifier, Info),
    ExtModule(Identifier, DefName, Parameters, Info),
    Port(Port),
    Stmt(Stmt),
    Type(Type),
    Expr(Expr),
    Info(Info),
}

/// Graph representation of FIRRTL AST for GumTree algorithm
#[derive(Debug, Clone)]
pub struct FirrtlGraph {
    pub graph: Graph<FirrtlNode, OrderedEdge>,
    pub root: NodeIndex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedEdge {
    /// Index representing the order of this edge among its siblings
    order: usize,
    /// Type of the edge to help with reconstruction
    edge_type: EdgeType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EdgeType {
    Port,
    Statement,
    Other,
}

#[derive(Debug)]
pub enum FirrtlGraphError {
    NodeNotFound(NodeIndex),
    InvalidNodeType(String),
    MissingChild(String),
    GraphMismatch(String),
}

impl FirrtlGraph {
    pub fn from_circuit(circuit: &Circuit) -> Self {
        let mut graph: Graph<FirrtlNode, OrderedEdge> = Graph::new();
        let root = graph.add_node(FirrtlNode::Circuit(
                circuit.version.clone(),
                circuit.name.clone(),
                circuit.annos.clone()));

        // Build graph recursively
        for (module_idx, module) in circuit.modules.iter().enumerate() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    let idx = graph.add_node(
                        FirrtlNode::Module(
                            m.name.clone(),
                            m.info.clone()));
                    graph.add_edge(root, idx, OrderedEdge { order: module_idx, edge_type: EdgeType::Other });
                    Self::add_ports(&mut graph, idx, &m.ports);
                    Self::add_statements(&mut graph, idx, &m.stmts);
                }
                CircuitModule::ExtModule(m) => {
                    let idx = graph.add_node(
                        FirrtlNode::ExtModule(
                            m.name.clone(),
                            m.defname.clone(),
                            m.params.clone(),
                            m.info.clone()));
                    graph.add_edge(root, idx, OrderedEdge { order: module_idx, edge_type: EdgeType::Other });
                    Self::add_ports(&mut graph, idx, &m.ports);
                }
            }
        }
        Self { graph, root }
    }

    fn add_ports(graph: &mut Graph<FirrtlNode, OrderedEdge>, parent: NodeIndex, ports: &Ports) {
        for (port_idx, port) in ports.iter().enumerate() {
            let idx = graph.add_node(FirrtlNode::Port(port.as_ref().clone()));
            graph.add_edge(parent, idx, OrderedEdge { order: port_idx, edge_type: EdgeType::Port });
        }
    }

    fn add_statements(graph: &mut Graph<FirrtlNode, OrderedEdge>, parent: NodeIndex, stmts: &Stmts) {
        for (stmt_idx, stmt) in stmts.iter().enumerate() {
            let idx = graph.add_node(FirrtlNode::Stmt(stmt.as_ref().clone()));
            graph.add_edge(parent, idx, OrderedEdge { order: stmt_idx, edge_type: EdgeType::Statement });

            match stmt.as_ref() {
                Stmt::Skip(_) => {}, // No additional nodes needed for Skip
                Stmt::Wire(_, tpe, _) => {
                    Self::add_tpe(graph, idx, tpe);
                }
                Stmt::Reg(_, tpe, clk, _) => {
                    Self::add_tpe(graph, idx, tpe);
                    Self::add_expression(graph, idx, clk);
                }
                Stmt::RegReset(_, tpe, clk, rst, init, _) => {
                    Self::add_tpe(graph, idx, tpe);
                    Self::add_expression(graph, idx, clk);
                    Self::add_expression(graph, idx, rst);
                    Self::add_expression(graph, idx, init);
                }
                Stmt::ChirrtlMemory(mem) => {
                    match mem {
                        ChirrtlMemory::SMem(_name, tpe, _, _) |
                        ChirrtlMemory::CMem(_name, tpe, _) => {
                            Self::add_tpe(graph, idx, tpe);
                        }
                    }
                }
                Stmt::ChirrtlMemoryPort(port) => {
                    match port {
                        ChirrtlMemoryPort::Write(name, mem, addr, clk, _) |
                        ChirrtlMemoryPort::Read(name, mem, addr, clk, _)  |
                        ChirrtlMemoryPort::Infer(name, mem, addr, clk, _) => {
                            Self::add_expression(graph, idx, &Expr::Reference(Reference::Ref(name.clone())));
                            Self::add_expression(graph, idx, &Expr::Reference(Reference::Ref(mem.clone())));
                            Self::add_expression(graph, idx, addr);
                            Self::add_expression(graph, idx, &Expr::Reference(clk.clone()));
                        }
                    }
                }
                Stmt::Inst(..) => {}
                Stmt::Node(_id, expr, _) => {
                    Self::add_expression(graph, idx, expr);
                }
                Stmt::Connect(lhs, rhs, _) => {
                    Self::add_expression(graph, idx, lhs);
                    Self::add_expression(graph, idx, rhs);
                }
                Stmt::When(pred, _, when, else_opt) => {
                    Self::add_expression(graph, idx, pred);

                    Self::add_statements(graph, idx, when);

                    if let Some(else_stmts) = else_opt {
                        Self::add_statements(graph, idx, else_stmts);
                    }
                }
                Stmt::Invalidate(lhs, _) => {
                    Self::add_expression(graph, idx, lhs);
                }
                Stmt::Printf(clock, en, _msg, args_opt, _) => {
                    // Add clock and enable expressions
                    Self::add_expression(graph, idx, clock);
                    Self::add_expression(graph, idx, en);

                    // Add all argument expressions
                    if let Some(args) = args_opt {
                        for arg in args {
                            Self::add_expression(graph, idx, arg);
                        }
                    }
                }
                Stmt::Assert(clock, pred, en, _msg, _) => {
                    // Add clock, predicate and enable expressions
                    Self::add_expression(graph, idx, clock);
                    Self::add_expression(graph, idx, pred);
                    Self::add_expression(graph, idx, en);
                }
            }
        }
    }

    fn add_expression(graph: &mut Graph<FirrtlNode, OrderedEdge>, parent: NodeIndex, expr: &Expr) {
        let idx = graph.add_node(FirrtlNode::Expr(expr.clone()));
        graph.add_edge(parent, idx, OrderedEdge { order: 0, edge_type: EdgeType::Other });

        match expr {
            Expr::Mux(cond, tval, fval) => {
                Self::add_expression(graph, idx, cond);
                Self::add_expression(graph, idx, tval);
                Self::add_expression(graph, idx, fval);
            }
            Expr::ValidIf(cond, value) => {
                Self::add_expression(graph, idx, cond);
                Self::add_expression(graph, idx, value);
            }
            Expr::PrimOp2Expr(_op, lhs, rhs) => {
                Self::add_expression(graph, idx, lhs);
                Self::add_expression(graph, idx, rhs);
            }
            Expr::PrimOp1Expr(_, arg) |
            Expr::PrimOp1Expr1Int(_, arg, ..) |
            Expr::PrimOp1Expr2Int(_, arg, ..) => {
                Self::add_expression(graph, idx, arg);
            }
            _ => {
            }
        }
    }

    fn add_tpe(graph: &mut Graph<FirrtlNode, OrderedEdge>, parent: NodeIndex, tpe: &Type) {
        let idx = graph.add_node(FirrtlNode::Type(tpe.clone()));
        graph.add_edge(parent, idx, OrderedEdge { order: 0, edge_type: EdgeType::Other });
    }

    /// Reconstruct a Circuit from the graph representation
    pub fn to_circuit(&self) -> Result<Circuit, FirrtlGraphError> {
        match &self.graph[self.root] {
            FirrtlNode::Circuit(version, name, annos) => {
                let mut modules = Vec::new();

                let mut module_edges: Vec<_> = self.graph.edges(self.root).collect();
                module_edges.sort_by_key(|edge| edge.weight().order);

                for edge in module_edges {
                    let dst = self.graph.edge_endpoints(edge.id()).unwrap().1;
                    match &self.graph[dst] {
                        FirrtlNode::Module(..) => {
                            let module = self.reconstruct_module(dst)?;
                            modules.push(Box::new(CircuitModule::Module(module)));
                        }
                        FirrtlNode::ExtModule(..) => {
                            let ext_module = self.reconstruct_ext_module(dst)?;
                            modules.push(Box::new(CircuitModule::ExtModule(ext_module)));
                        }
                        _ => return Err(FirrtlGraphError::InvalidNodeType("Expected Module or ExtModule".to_string())),
                    }
                }

                Ok(Circuit {
                    version: version.clone(),
                    name: name.clone(),
                    annos: annos.clone(),
                    modules,
                })
            }
            _ => Err(FirrtlGraphError::InvalidNodeType("Root node must be Circuit".to_string())),
        }
    }

    fn reconstruct_module(&self, module_idx: NodeIndex) -> Result<Module, FirrtlGraphError> {
        match &self.graph[module_idx] {
            FirrtlNode::Module(name, info) => {
                let mut ports = Vec::new();
                let mut stmts = Vec::new();

                // Get all edges and sort by order
                let mut port_edges: Vec<_> = self.graph.edges(module_idx)
                    .filter(|edge| edge.weight().edge_type == EdgeType::Port)
                    .collect();
                port_edges.sort_by_key(|edge| edge.weight().order);

                let mut stmt_edges: Vec<_> = self.graph.edges(module_idx)
                    .filter(|edge| edge.weight().edge_type == EdgeType::Statement)
                    .collect();
                stmt_edges.sort_by_key(|edge| edge.weight().order);

                // Process ports in order
                for edge in port_edges {
                    let dst = self.graph.edge_endpoints(edge.id()).unwrap().1;
                    ports.push(self.reconstruct_port(dst)?)
                }

                // Process statements in order
                for edge in stmt_edges {
                    let dst = self.graph.edge_endpoints(edge.id()).unwrap().1;
                    stmts.push(self.reconstruct_stmt(dst)?)
                }

                Ok(Module {
                    name: name.clone(),
                    ports,
                    stmts,
                    info: info.clone(),
                })
            }
            _ => Err(FirrtlGraphError::InvalidNodeType("Expected Module".to_string())),
        }
    }

    fn reconstruct_ext_module(&self, module_idx: NodeIndex) -> Result<ExtModule, FirrtlGraphError> {
        match &self.graph[module_idx] {
            FirrtlNode::ExtModule(name, defname, params, info) => {
                let mut ports = Vec::new();

                // Get all port edges and sort by order
                let mut port_edges: Vec<_> = self.graph.edges(module_idx)
                    .filter(|edge| edge.weight().edge_type == EdgeType::Port)
                    .collect();
                port_edges.sort_by_key(|edge| edge.weight().order);

                // Process ports in order
                for edge in port_edges {
                    let dst = self.graph.edge_endpoints(edge.id()).unwrap().1;
                    ports.push(self.reconstruct_port(dst)?)
                }

                Ok(ExtModule {
                    name: name.clone(),
                    ports,
                    defname: defname.clone(),
                    params: params.clone(),
                    info: info.clone(),
                })
            }
            _ => Err(FirrtlGraphError::InvalidNodeType("Expected ExtModule".to_string())),
        }
    }

    fn reconstruct_port(&self, port_idx: NodeIndex) -> Result<Box<Port>, FirrtlGraphError> {
        match &self.graph[port_idx] {
            FirrtlNode::Port(port) => {
                Ok(Box::new(port.clone()))
            }
            _ => Err(FirrtlGraphError::InvalidNodeType("Expected Port".to_string())),
        }
    }

    fn reconstruct_stmt(&self, stmt_idx: NodeIndex) -> Result<Box<Stmt>, FirrtlGraphError> {
        match &self.graph[stmt_idx] {
            FirrtlNode::Stmt(stmt) => {
                Ok(Box::new(stmt.clone()))
            }
            _ => Err(FirrtlGraphError::InvalidNodeType("Expected Stmt".to_string())),
        }
    }

    /// Verify that the graph can be correctly reconstructed into an equivalent AST
    pub fn verify(&self, original: &Circuit) -> Result<(), FirrtlGraphError> {
        let reconstructed = self.to_circuit()?;
        if reconstructed == *original {
            Ok(())
        } else {
            let mut printer = Printer::new();
            let circuit_str = printer.print_circuit(&reconstructed);
            std::fs::write(
                &format!("./test-outputs/{}.reconstruct.fir", original.name.to_string()),
                circuit_str
            ).expect("to work");

            Err(FirrtlGraphError::GraphMismatch(
                    "Reconstructed AST does not match original".to_string()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;
    use chirrtl_parser::parse_circuit;

    #[test_case("GCD" ; "GCD")]
    #[test_case("chipyard.harness.TestHarness.RocketConfig" ; "Rocket")]
    fn test_graph_reconstruction(name: &str) -> Result<(), FirrtlGraphError> {
        let source = std::fs::read_to_string(format!("./test-inputs/{}.fir", name)).expect("to_exist");
        let circuit = parse_circuit(&source).expect("firrtl parser");

        // Create a simple test circuit
        // Convert to graph
        let graph = FirrtlGraph::from_circuit(&circuit);

        // Verify reconstruction
        graph.verify(&circuit)
    }
}
