use chirrtl_parser::ast::*;
use petgraph::{
    graph::{NodeIndex, Graph},
    visit::{VisitMap, Visitable},
};


/// A node in the FIRRTL AST for GumTree comparison
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FirrtlNode {
    Circuit(Circuit),
    Module(Module),
    ExtModule(ExtModule),
    Port(Port),
    Stmt(Stmt),
    Type(Type),
    Expr(Expr),
    Reference(Reference),
    Info(Info),
}

/// Graph representation of FIRRTL AST for GumTree algorithm
#[derive(Debug, Clone)]
pub struct FirrtlGraph {
    pub graph: Graph<FirrtlNode, ()>,
    pub root: NodeIndex,
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
        let mut graph = Graph::new();
        let root = graph.add_node(FirrtlNode::Circuit(circuit.clone()));

        // Build graph recursively
        for module in circuit.modules.iter() {
            let module_idx = match module.as_ref() {
                CircuitModule::Module(m) => {
                    let idx = graph.add_node(FirrtlNode::Module(m.clone()));
                    graph.add_edge(root, idx, ());

                    // Add ports
                    Self::add_ports(&mut graph, idx, &m.ports);

                    // Add statements
                    Self::add_statements(&mut graph, idx, &m.stmts);

                    idx
                }
                CircuitModule::ExtModule(e) => {
                    let idx = graph.add_node(FirrtlNode::ExtModule(e.clone()));
                    graph.add_edge(root, idx, ());

                    // Add ports for external module
                    Self::add_ports(&mut graph, idx, &e.ports);

                    idx
                }
            };
        }
        Self { graph, root }
    }

    fn add_ports(graph: &mut Graph<FirrtlNode, ()>, parent: NodeIndex, ports: &Ports) {
        for port in ports.iter() {
            let port_idx = graph.add_node(FirrtlNode::Port(port.as_ref().clone()));
            graph.add_edge(parent, port_idx, ());

            // Add port type
            match port.as_ref() {
                Port::Input(_, tpe, _) | Port::Output(_, tpe, _) => {
                    let type_idx = graph.add_node(FirrtlNode::Type(tpe.clone()));
                    graph.add_edge(port_idx, type_idx, ());
                }
            }
        }
    }

    fn add_statements(graph: &mut Graph<FirrtlNode, ()>, parent: NodeIndex, stmts: &Stmts) {
        for stmt in stmts.iter() {
            let stmt_idx = graph.add_node(FirrtlNode::Stmt(stmt.as_ref().clone()));
            graph.add_edge(parent, stmt_idx, ());

            match stmt.as_ref() {
                Stmt::Node(id, expr, _) => {
                    Self::add_expression(graph, stmt_idx, expr);
                }
                Stmt::Connect(ref_, expr, _) => {
                    let ref_idx = graph.add_node(FirrtlNode::Reference(ref_.clone()));
                    graph.add_edge(stmt_idx, ref_idx, ());
                    Self::add_expression(graph, stmt_idx, expr);
                }
                Stmt::Conditionally(pred, conseq, alt, _) => {
                    Self::add_expression(graph, stmt_idx, pred);
                    Self::add_statements(graph, stmt_idx, conseq);
                    Self::add_statements(graph, stmt_idx, alt);
                }
                Stmt::ChirrtlMemory(id, tpe, size, ruw, _) => {
                    let type_idx = graph.add_node(FirrtlNode::Type(tpe.clone()));
                    graph.add_edge(stmt_idx, type_idx, ());
                }
                Stmt::ChirrtlMemoryPort(port, _) => {
                    match port {
                        ChirrtlMemoryPort::Write(_, _, addr, data, _) |
                        ChirrtlMemoryPort::Read(_, _, addr, _, _) |
                        ChirrtlMemoryPort::ReadWrite(_, _, addr, _, _, _) => {
                            Self::add_expression(graph, stmt_idx, addr);
                            if let ChirrtlMemoryPort::Write(_, _, _, data, _) = port {
                                let ref_idx = graph.add_node(FirrtlNode::Reference(data.clone()));
                                graph.add_edge(stmt_idx, ref_idx, ());
                            }
                        }
                    }
                }
                // Add other statement types as needed
                _ => {}
            }
        }
    }

    fn add_expression(graph: &mut Graph<FirrtlNode, ()>, parent: NodeIndex, expr: &Expr) {
        let expr_idx = graph.add_node(FirrtlNode::Expr(expr.clone()));
        graph.add_edge(parent, expr_idx, ());

        match expr {
            Expr::Reference(ref_) => {
                let ref_idx = graph.add_node(FirrtlNode::Reference(ref_.clone()));
                graph.add_edge(expr_idx, ref_idx, ());
            }
            Expr::SubField(ref_, field) => {
                let ref_idx = graph.add_node(FirrtlNode::Reference(ref_.clone()));
                graph.add_edge(expr_idx, ref_idx, ());
            }
            Expr::SubIndex(ref_, idx) => {
                let ref_idx = graph.add_node(FirrtlNode::Reference(ref_.clone()));
                graph.add_edge(expr_idx, ref_idx, ());
            }
            Expr::SubAccess(ref_, idx) => {
                let ref_idx = graph.add_node(FirrtlNode::Reference(ref_.clone()));
                graph.add_edge(expr_idx, ref_idx, ());
                Self::add_expression(graph, expr_idx, idx);
            }
            Expr::Mux(cond, tval, fval, _) => {
                Self::add_expression(graph, expr_idx, cond);
                Self::add_expression(graph, expr_idx, tval);
                Self::add_expression(graph, expr_idx, fval);
            }
            Expr::ValidIf(cond, value, _) => {
                Self::add_expression(graph, expr_idx, cond);
                Self::add_expression(graph, expr_idx, value);
            }
            Expr::PrimOp2(op, lhs, rhs, _) => {
                Self::add_expression(graph, expr_idx, lhs);
                Self::add_expression(graph, expr_idx, rhs);
            }
            Expr::PrimOp1(op, arg, _) => {
                Self::add_expression(graph, expr_idx, arg);
            }
            // Add other expression types as needed
            _ => {}
        }
    }

    /// Reconstruct a Circuit from the graph representation
    pub fn to_circuit(&self) -> Result<Circuit, FirrtlGraphError> {
        match &self.graph[self.root] {
            FirrtlNode::Circuit(circuit) => {
                let mut modules = Vec::new();
                for module_idx in self.graph.neighbors(self.root) {
                    match &self.graph[module_idx] {
                        FirrtlNode::Module(_) => {
                            let module = self.reconstruct_module(module_idx)?;
                            modules.push(Box::new(CircuitModule::Module(module)));
                        }
                        FirrtlNode::ExtModule(_) => {
                            let ext_module = self.reconstruct_ext_module(module_idx)?;
                            modules.push(Box::new(CircuitModule::ExtModule(ext_module)));
                        }
                        _ => return Err(FirrtlGraphError::InvalidNodeType("Expected Module or ExtModule".to_string())),
                    }
                }
                Ok(Circuit {
                    version: circuit.version.clone(),
                    name: circuit.name.clone(),
                    annos: circuit.annos.clone(),
                    modules,
                })
            }
            _ => Err(FirrtlGraphError::InvalidNodeType("Root node must be Circuit".to_string())),
        }
    }

    fn reconstruct_module(&self, module_idx: NodeIndex) -> Result<Module, FirrtlGraphError> {
        match &self.graph[module_idx] {
            FirrtlNode::Module(module) => {
                let mut ports = Vec::new();
                let mut stmts = Vec::new();

                for child_idx in self.graph.neighbors(module_idx) {
                    match &self.graph[child_idx] {
                        FirrtlNode::Port(_) => {
                            ports.push(self.reconstruct_port(child_idx)?);                            
                        }
                        FirrtlNode::Stmt(_) => {
                            stmts.push(self.reconstruct_stmt(child_idx)?);                            
                        }
                        _ => {}
                    }
                }

                Ok(Module {
                    name: module.name.clone(),
                    ports,
                    stmts,
                    info: module.info.clone(),
                })
            }
            _ => Err(FirrtlGraphError::InvalidNodeType("Expected Module".to_string())),
        }
    }

    fn reconstruct_ext_module(&self, module_idx: NodeIndex) -> Result<ExtModule, FirrtlGraphError> {
        match &self.graph[module_idx] {
            FirrtlNode::ExtModule(module) => {
                let mut ports = Vec::new();

                for child_idx in self.graph.neighbors(module_idx) {
                    match &self.graph[child_idx] {
                        FirrtlNode::Port(_) => {
                            ports.push(self.reconstruct_port(child_idx)?);
                        }
                        _ => {}
                    }
                }

                Ok(ExtModule {
                    name: module.name.clone(),
                    ports,
                    defname: module.defname.clone(),
                    params: module.params.clone(),
                    info: module.info.clone(),
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
            Err(FirrtlGraphError::GraphMismatch(
                "Reconstructed AST does not match original".to_string()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_reconstruction() -> Result<(), FirrtlGraphError> {
        // Create a simple test circuit
        let circuit = Circuit {
            version: Version::default(),
            name: Identifier::Name("test".to_string()),
            annos: Annotations::default(),
            modules: vec![Box::new(CircuitModule::Module(Module {
                name: Identifier::Name("TestModule".to_string()),
                ports: vec![Box::new(Port::Input(
                    Identifier::Name("in".to_string()),
                    Type::TypeGround(TypeGround::UInt(Width(1))),
                    Info::default(),
                ))],
                stmts: vec![Box::new(Stmt::Node(
                    Identifier::Name("node".to_string()),
                    Box::new(Expr::Reference(Reference::Ref(
                        Identifier::Name("in".to_string())
                    ))),
                    Info::default(),
                ))],
                info: Info::default(),
            }))],
        };
        // Convert to graph
        let graph = FirrtlGraph::from_circuit(&circuit);
        // Verify reconstruction
        graph.verify(&circuit)
    }

    #[test]
    fn test_module_reconstruction() -> Result<(), FirrtlGraphError> {
        // Create a test module with various statement types
        let module = Module {
            name: Identifier::Name("TestModule".to_string()),
            ports: vec![
                Box::new(Port::Input(
                    Identifier::Name("in1".to_string()),
                    Type::TypeGround(TypeGround::UInt(Width(1))),
                    Info::default(),
                )),
                Box::new(Port::Output(
                    Identifier::Name("out1".to_string()),
                    Type::TypeGround(TypeGround::UInt(Width(1))),
                    Info::default(),
                )),
            ],
            stmts: vec![
                Box::new(Stmt::Node(
                    Identifier::Name("n1".to_string()),
                    Box::new(Expr::Reference(Reference::Ref(
                        Identifier::Name("in1".to_string())
                    ))),
                    Info::default(),
                )),
                Box::new(Stmt::Connect(
                    Reference::Ref(Identifier::Name("out1".to_string())),
                    Box::new(Expr::Reference(Reference::Ref(
                        Identifier::Name("n1".to_string())
                    ))),
                    Info::default(),
                )),
            ],
            info: Info::default(),
        };

        // Create a circuit containing this module
        let circuit = Circuit {
            version: Version::default(),
            name: Identifier::Name("test".to_string()),
            annos: Annotations::default(),
            modules: vec![Box::new(CircuitModule::Module(module))],
        };

        // Convert to graph and verify
        let graph = FirrtlGraph::from_circuit(&circuit);
        graph.verify(&circuit)
    }
}
