use chirrtl_parser::ast::*;
use petgraph::{graph::{Graph, NodeIndex}, visit::EdgeRef, Direction::Outgoing};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::cmp::max;
use std::fmt::Display;
use indexmap::IndexMap;
use crate::passes::ast::print::Printer;

/// A node in the FIRRTL AST for GumTree comparison
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ASTElement {
    Circuit(Version, Identifier, Annotations),
    Module(Identifier, Info),
    ExtModule(Identifier, DefName, Parameters, Info),
    Port(Port),
    Stmt(Stmt),
    Type(Type),
    Expr(Expr),
}

impl std::hash::Hash for ASTElement {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash discriminant first to differentiate between variants
        std::mem::discriminant(self).hash(state);

        match self {
            // For Circuit, only hash Version and Identifier, ignore Annotations
            ASTElement::Circuit(version, id, _) => {
                version.hash(state);
                id.hash(state);
            }
            // For Module, only hash Identifier, ignore Info
            ASTElement::Module(id, _) => {
                id.hash(state);
            }
            // For ExtModule, hash Identifier and DefName, ignore Parameters and Info
            ASTElement::ExtModule(id, defname, _, _) => {
                id.hash(state);
                defname.hash(state);
            }
            // For Port, Type, Stmt, and Expr, use their default Hash implementation
            ASTElement::Port(port) => {
                std::mem::discriminant(port).hash(state);
                match port {
                    Port::Input(name, tpe, _) |
                        Port::Output(name, tpe, _) => {
                        name.hash(state);
                        tpe.hash(state);
                    }
                }
            }
            ASTElement::Stmt(stmt) => {
                std::mem::discriminant(stmt).hash(state);
                match stmt {
                    Stmt::Wire(name, _, _)       |
                        Stmt::Reg(name, ..)      |
                        Stmt::RegReset(name, ..) |
                        Stmt::Inst(name, ..)     |
                        Stmt::Node(name, ..)     => {
                        name.hash(state);
                    }
                    Stmt::Skip(_info) => {}
                    Stmt::Connect(_lhs, _rhs, _info) => {}
                    Stmt::Invalidate(_lhs, _info) => {}
                    Stmt::When(_cond, _info, _when, _else_opt) => {}
                    Stmt::Printf(_name, _clk, _en, msg, _args, _info) => {
                        msg.hash(state);
                    }
                    Stmt::Assert(_name, _clk, _en, _cond, msg, _info) => {
                        msg.hash(state);
                    }
                    Stmt::ChirrtlMemory(..) => {}
                    Stmt::ChirrtlMemoryPort(..) => {}
                }
            }
            ASTElement::Type(typ) => typ.hash(state),
            ASTElement::Expr(expr) => expr.hash(state),
        }
    }
}

pub type HashVal = u64;
pub type Height = u32;
pub type Descs = u32;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FirrtlNode {
    pub elem: ASTElement,
    pub height: Height,
    pub hash: HashVal,
    pub descs: Descs,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ASTNodeLabel(String);

impl From<ASTElement> for FirrtlNode {
    fn from(value: ASTElement) -> Self {
        FirrtlNode { elem: value, height: 0, hash: 0, descs: 0 }
    }
}

impl FirrtlNode {
    pub fn label(&self) -> ASTNodeLabel {
        match &self.elem {
            ASTElement::Circuit(_, name, _)  |
                ASTElement::Module(name, ..) |
                ASTElement::ExtModule(name, ..) => {
                    ASTNodeLabel(name.to_string())
            }
            ASTElement::Port(port) => {
                ASTNodeLabel(port.to_string())
            }
            ASTElement::Stmt(stmt) => {
                ASTNodeLabel(stmt.to_string())
            }
            ASTElement::Type(tpe) => {
                ASTNodeLabel(tpe.to_string())
            }
            ASTElement::Expr(expr) => {
                ASTNodeLabel(expr.to_string())
            }
        }
    }
}

impl Display for FirrtlNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.height, self.label().0.replace('"', ""))
    }
}

/// Graph representation of FIRRTL AST for GumTree algorithm
#[derive(Debug, Clone)]
pub struct FirrtlGraph {
    pub graph: Graph<FirrtlNode, OrderedEdge>,
    pub root: NodeIndex,
}

/// To maintain the order of statements when reconstructing the AST
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
    /// Returns the height of the node
    pub fn height(&self, id: NodeIndex) -> Height {
        self.graph.node_weight(id).unwrap().height
    }

    /// Compute the height and hash of each node
    pub fn compute_metadata(&mut self) {
        // We can merge these three functions into one for performance...
        self.compute_height();
        self.compute_hash();
        self.compute_num_desc();
    }

    fn compute_num_desc(&mut self) {
         let mut dfs = petgraph::visit::DfsPostOrder::new(&self.graph, self.root);

        while let Some(nidx) = dfs.next(&self.graph) {
            let childs = self.graph.neighbors_directed(nidx, Outgoing);
            let mut num_desc = 0;

            for cidx in childs {
                let cnode =  self.graph.node_weight(cidx).unwrap();
                num_desc += cnode.descs + 1;
            }
            let node = self.graph.node_weight_mut(nidx).unwrap();
            node.descs = num_desc;
        }
    }

    /// Compute height of each tree node
    fn compute_height(&mut self) {
        let mut dfs = petgraph::visit::DfsPostOrder::new(&self.graph, self.root);
        while let Some(id) = dfs.next(&self.graph) {
            let is_leaf = self.graph.neighbors_directed(id, Outgoing).count() == 0;
            if is_leaf {
                let node = self.graph.node_weight_mut(id).unwrap();
                node.height = 1;
            } else {
                let mut max_child_height = 0;
                for cid in self.graph.neighbors_directed(id, Outgoing) {
                    let child = self.graph.node_weight(cid).unwrap();
                    max_child_height = max(max_child_height, child.height);
                }
                let node = self.graph.node_weight_mut(id).unwrap();
                node.height = max_child_height + 1;
            }
        }
    }

    /// Compute hash of each tree node
    fn compute_hash(&mut self) {
        let mut dfs = petgraph::visit::DfsPostOrder::new(&self.graph, self.root);
        while let Some(id) = dfs.next(&self.graph) {
            let mut hasher = DefaultHasher::new();
            let is_leaf = self.graph.neighbors_directed(id, Outgoing).count() == 0;
            if is_leaf {
                let node = self.graph.node_weight_mut(id).unwrap();

                // Super naive hasing
                node.elem.hash(&mut hasher);
                node.hash = hasher.finish();
            } else {
                // Process all children in order
                let mut ordered_edges: Vec<_> = self.graph.edges(id).collect();
                ordered_edges.sort_by_key(|e| e.weight().order);
                for edge in ordered_edges {
                    let cid = self.graph.edge_endpoints(edge.id()).unwrap().1;
                    let child = self.graph.node_weight(cid).unwrap();
                    child.hash.hash(&mut hasher);
                }

                let node = self.graph.node_weight_mut(id).unwrap();
                node.elem.hash(&mut hasher);
                node.hash = hasher.finish();
            }
        }
    }

     /// Returns a set containing all node hash values
    pub fn hash_count(self: &Self) -> IndexMap<HashVal, u32> {
        let mut ret: IndexMap<HashVal, u32> = IndexMap::new();
        for nw in self.graph.node_weights() {
            if !ret.contains_key(&nw.hash) {
                ret.insert(nw.hash, 0);
            }
            let cnt = ret.get_mut(&nw.hash).unwrap();
            *cnt += 1;
        }
        return ret;
    }

    /// Returns a map from node labels to a list of node indices with the corresponding label
    pub fn labels(self: &Self) -> IndexMap<ASTNodeLabel, Vec<NodeIndex>> {
        let mut ret = IndexMap::new();
        for nidx in self.graph.node_indices() {
            let node = self.graph.node_weight(nidx).unwrap();
            let label = node.label();
            if !ret.contains_key(&label) {
                ret.insert(label.clone(), vec![]);
            }
            ret.get_mut(&label).unwrap().push(nidx);
        }
        return ret;
    }
}

impl FirrtlGraph {
    pub fn from_circuit(circuit: &Circuit) -> Self {
        let mut graph: Graph<FirrtlNode, OrderedEdge> = Graph::new();
        let root = graph.add_node(FirrtlNode::from(ASTElement::Circuit(
                circuit.version.clone(),
                circuit.name.clone(),
                circuit.annos.clone())));

        // Build graph recursively
        for (module_idx, module) in circuit.modules.iter().enumerate() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    let idx = graph.add_node(FirrtlNode::from(
                        ASTElement::Module(
                            m.name.clone(),
                            m.info.clone())));
                    graph.add_edge(root, idx, OrderedEdge { order: module_idx, edge_type: EdgeType::Other });
                    Self::add_ports(&mut graph, idx, &m.ports);
                    Self::add_statements(&mut graph, idx, &m.stmts);
                }
                CircuitModule::ExtModule(m) => {
                    let idx = graph.add_node(FirrtlNode::from(
                        ASTElement::ExtModule(
                            m.name.clone(),
                            m.defname.clone(),
                            m.params.clone(),
                            m.info.clone())));
                    graph.add_edge(root, idx, OrderedEdge { order: module_idx, edge_type: EdgeType::Other });
                    Self::add_ports(&mut graph, idx, &m.ports);
                }
            }
        }

        let mut ret = Self { graph, root };
        ret.compute_metadata();
        ret
    }

    fn add_ports(graph: &mut Graph<FirrtlNode, OrderedEdge>, parent: NodeIndex, ports: &Ports) {
        for (port_idx, port) in ports.iter().enumerate() {
            let idx = graph.add_node(FirrtlNode::from(ASTElement::Port(port.as_ref().clone())));
            graph.add_edge(parent, idx, OrderedEdge { order: port_idx, edge_type: EdgeType::Port });
        }
    }

    fn add_statements(graph: &mut Graph<FirrtlNode, OrderedEdge>, parent: NodeIndex, stmts: &Stmts) {
        for (stmt_idx, stmt) in stmts.iter().enumerate() {
            let idx = graph.add_node(FirrtlNode::from(ASTElement::Stmt(stmt.as_ref().clone())));
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
                Stmt::Printf(_name, clock, en, _msg, args_opt, _) => {
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
                Stmt::Assert(_name, clock, pred, en, _msg, _) => {
                    // Add clock, predicate and enable expressions
                    Self::add_expression(graph, idx, clock);
                    Self::add_expression(graph, idx, pred);
                    Self::add_expression(graph, idx, en);
                }
            }
        }
    }

    fn add_expression(graph: &mut Graph<FirrtlNode, OrderedEdge>, parent: NodeIndex, expr: &Expr) {
        let idx = graph.add_node(FirrtlNode::from(ASTElement::Expr(expr.clone())));
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
        let idx = graph.add_node(FirrtlNode::from(ASTElement::Type(tpe.clone())));
        graph.add_edge(parent, idx, OrderedEdge { order: 0, edge_type: EdgeType::Other });
    }
}

impl FirrtlGraph {
    /// Reconstruct a Circuit from the graph representation
    pub fn to_circuit(&self) -> Result<Circuit, FirrtlGraphError> {
        match &self.graph[self.root].elem {
            ASTElement::Circuit(version, name, annos) => {
                let mut modules = Vec::new();

                let mut module_edges: Vec<_> = self.graph.edges(self.root).collect();
                module_edges.sort_by_key(|edge| edge.weight().order);

                for edge in module_edges {
                    let dst = self.graph.edge_endpoints(edge.id()).unwrap().1;
                    match &self.graph[dst].elem {
                        ASTElement::Module(..) => {
                            let module = self.reconstruct_module(dst)?;
                            modules.push(Box::new(CircuitModule::Module(module)));
                        }
                        ASTElement::ExtModule(..) => {
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
        match &self.graph[module_idx].elem {
            ASTElement::Module(name, info) => {
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
        match &self.graph[module_idx].elem {
            ASTElement::ExtModule(name, defname, params, info) => {
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
        match &self.graph[port_idx].elem {
            ASTElement::Port(port) => {
                Ok(Box::new(port.clone()))
            }
            _ => Err(FirrtlGraphError::InvalidNodeType("Expected Port".to_string())),
        }
    }

    fn reconstruct_stmt(&self, stmt_idx: NodeIndex) -> Result<Box<Stmt>, FirrtlGraphError> {
        match &self.graph[stmt_idx].elem {
            ASTElement::Stmt(stmt) => {
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
