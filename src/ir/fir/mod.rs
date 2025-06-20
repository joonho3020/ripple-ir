//! FIRRTL Intermediate Representation (FIR) graph structures and utilities.
//!
//! This module defines the core data structures for representing FIRRTL circuits as graphs,
//! including nodes for operations, registers, memories, and ports, as well as edges for dataflow and control.
//! It also provides helper methods for graph construction and manipulation, and utilities for type inference and visualization.

use petgraph::graph::EdgeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction::Incoming;
use petgraph::Direction::Outgoing;
use rusty_firrtl::*;
use derivative::Derivative;
use indexmap::IndexMap;
use indexmap::IndexSet;
use petgraph::graph::{Graph, NodeIndex};
use crate::common::graphviz::*;
use crate::common::RippleIRErr;
use crate::ir::typetree::typetree::*;
use crate::impl_clean_display;

use super::hierarchy::Hierarchy;
use super::whentree::CondPathWithPrior;

/// A node in the FIRRTL IR graph, representing an operation, register, memory, port, etc.
#[derive(Derivative, Clone)]
#[derivative(Debug)]
pub struct FirNode {
    /// Optional name of the node (signal, register, etc.)
    pub name: Option<Identifier>,

    /// The type of node (operation, register, etc.)
    pub nt: FirNodeType,

    /// Optional type tree for this node (signal type information)
    #[derivative(Debug="ignore")]
    pub ttree: Option<TypeTree>,
}

impl FirNode {
    /// Create a new FIR node with the given name, type, and optional type tree.
    pub fn new(name: Option<Identifier>, nt: FirNodeType, ttree: Option<TypeTree>) -> Self {
        Self { name, nt, ttree }
    }

    /// Returns true if this node is a phi node.
    pub fn is_phi(&self) -> bool {
        match self.nt {
            FirNodeType::Phi(..) => {
                true
            }
            _ => {
                false
            }
        }
    }
}

/// The type of a node in the FIRRTL IR graph.
#[derive(Debug, Default, Clone, PartialEq, Hash)]
pub enum FirNodeType {
    #[default]
    Invalid,

    DontCare,

    UIntLiteral(Width, Int),
    SIntLiteral(Width, Int),

    Mux,
    PrimOp2Expr(PrimOp2Expr),
    PrimOp1Expr(PrimOp1Expr),
    PrimOp1Expr1Int(PrimOp1Expr1Int, Int),
    PrimOp1Expr2Int(PrimOp1Expr2Int, Int, Int),

    // Stmt
    Wire,
    Reg,
    RegReset,
    SMem(Option<ChirrtlMemoryReadUnderWrite>),
    CMem,
    WriteMemPort(CondPathWithPrior),
    ReadMemPort(CondPathWithPrior),
    InferMemPort(CondPathWithPrior),
    Inst(Identifier),
    Printf(Stmt, CondPathWithPrior),
    Assert(Stmt, CondPathWithPrior),
    Memory(u32, u32, u32, MemoryPorts, ChirrtlMemoryReadUnderWrite),

    // Port
    Input,
    Output,
    Phi(CondPathWithPrior),
}

impl_clean_display!(FirNode);

/// An edge in the FIRRTL IR graph, representing dataflow or control between nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct FirEdge {
    /// The source expression for this edge.
    pub src: Expr,
    /// The optional destination reference for this edge.
    pub dst: Option<Reference>,
    /// The type of edge (data, clock, reset, etc.)
    pub et: FirEdgeType
}

/// The type of an edge in the FIRRTL IR graph.
#[derive(Derivative, Clone, PartialEq, Hash)]
#[derivative(Debug)]
pub enum FirEdgeType {
    Wire,

    InitValue,

    Operand0,
    Operand1,

    MuxCond,
    MuxTrue,
    MuxFalse,

    Clock,
    Reset,
    DontCare,

    PhiInput(CondPathWithPrior, bool),
    PhiSel,
    PhiOut,

    MemPortEdge,
    MemPortAddr,
    MemPortEn,

    ArrayAddr,
}

impl_clean_display!(FirEdge);

impl FirEdge {
    pub fn new(src: Expr, dst: Option<Reference>, et: FirEdgeType) -> Self {
        Self { src, et, dst }
    }
}

type IRGraph = Graph<FirNode, FirEdge>;

#[derive(Debug, Clone)]
pub struct ExtModuleInfo {
    pub defname: DefName,
    pub params: Parameters,
}

impl From<&ExtModule> for ExtModuleInfo {
    fn from(value: &ExtModule) -> Self {
        ExtModuleInfo {
            defname: value.defname.clone(),
            params: value.params.clone()
        }
    }
}

// TODO: NameSpace should be part of the IR
#[derive(Debug, Clone)]
pub struct NameSpace {
    used: IndexSet<Identifier>,
    cntr: u32,
    pfx: String
}

impl NameSpace {
    pub fn new(fg: &FirGraph) -> Self {
        let mut used: IndexSet<Identifier> = IndexSet::new();
        for id in fg.graph.node_indices() {
            let node = fg.graph.node_weight(id).unwrap();
            if let Some(name) = &node.name {
                used.insert(name.clone());
            }
        }
        Self {
            used,
            cntr: 0,
            pfx: "_TMP".to_string(),
        }
    }

    pub fn from_used_set(used: IndexSet<Identifier>) -> Self {
        Self {
            used,
            cntr: 0,
            pfx: "_RUSTGEN".to_string(),
        }
    }

    pub fn next(&mut self) -> Identifier {
        loop {
            let candidate = Identifier::Name(format!("{}_{}", self.pfx, self.cntr));
            self.cntr += 1;
            if !self.used.contains(&candidate) {
                return candidate;
            }
        }
    }

    pub fn contains(&self, x: &Identifier) -> bool {
        self.used.contains(x)
    }
}

impl Default for NameSpace {
    fn default() -> Self {
        Self {
            used: IndexSet::new(),
            cntr: 0,
            pfx: "_TMP".to_string(),
        }
    }
}

/// The main FIRRTL IR graph structure, containing nodes and edges.
#[derive(Debug, Clone)]
pub struct FirGraph {
    /// The underlying petgraph graph structure.
    pub graph: IRGraph,
    /// Optional external module information (for blackboxes).
    pub ext_info: Option<ExtModuleInfo>,
    /// Whether this is a blackbox module.
    pub blackbox: bool,
    /// Namespace for generating unique names within this graph.
    pub namespace: NameSpace
}

impl FirGraph {
    /// Create a new, empty FIR graph.
    pub fn new(blackbox: bool) -> Self {
        Self {
            graph: IRGraph::new(),
            blackbox,
            ext_info: None,
            namespace: NameSpace::default()
        }
    }

    /// Returns a TypeTree of all its IO signals. Useful for handling instances.
    pub fn io_typetree(&self) -> TypeTree {
        let mut io_ttrees: IndexMap<&Identifier, &TypeTree> = IndexMap::new();
        for id in self.graph.node_indices() {
            let node = self.graph.node_weight(id).unwrap();
            if node.nt == FirNodeType::Input || node.nt == FirNodeType::Output {
                io_ttrees.insert(&node.name.as_ref().unwrap(), &node.ttree.as_ref().unwrap());
            }
        }
        TypeTree::merge_trees(io_ttrees)
    }

    /// Returns an `EdgeIndex` coming into the node that matches `et` if exists.
    pub fn parent_with_type(&self, id: NodeIndex, et: FirEdgeType) -> Option<EdgeIndex> {
        for eid in self.graph.edges_directed(id, Incoming) {
            let edge = self.graph.edge_weight(eid.id()).unwrap();
            if edge.et == et {
                return Some(eid.id());
            }
        }
        return None;
    }

    /// Returns an `EdgeIndex` going out of node that matches `et` if exists.
    pub fn childs_with_type(&self, id: NodeIndex, et: FirEdgeType) -> Vec<EdgeIndex> {
        let mut ret: Vec<EdgeIndex> = vec![];
        for eid in self.graph.edges_directed(id, Outgoing) {
            let edge = self.graph.edge_weight(eid.id()).unwrap();
            if edge.et == et {
                ret.push(eid.id());
            }
        }
        ret
    }

    /// Returns true if a edge contains bidirectional connections.
    pub fn bidirectional(&self, id: EdgeIndex) -> bool {
        let edge = self.graph.edge_weight(id).unwrap();
        let (src, _dst) = self.graph.edge_endpoints(id).unwrap();
        let ttree = self.graph.node_weight(src).unwrap()
            .ttree.as_ref().unwrap()
            .view().unwrap();

        let subtree = ttree.subtree_from_expr(&edge.src).unwrap();
        subtree.is_bidirectional()
    }

    /// Rebuilds the namespace for this graph.
    pub fn build_namespace(&mut self) {
        self.namespace = NameSpace::new(self);
    }

    /// Adds a new unsigned integer literal node to the graph.
    pub fn add_uint_literal(&mut self, value: u32, width: u32) -> (NodeIndex, Expr) {
        let node = FirNode::new(
            None,
            FirNodeType::UIntLiteral(Width(width), Int::from(value)),
            None,
        );
        let expr = Expr::UIntInit(Width(width), Int::from(value));
        (self.graph.add_node(node), expr)
    }

    /// Adds a new binary primitive operation node (e.g., Add, And, Eq) to the graph.
    pub fn add_primop2(
        &mut self,
        op: PrimOp2Expr,
        lhs_id: NodeIndex,
        lhs_expr: Expr,
        rhs_id: NodeIndex,
        rhs_expr: Expr,
    ) -> (NodeIndex, Expr) {
        let name = self.namespace.next();
        let node = FirNode::new(Some(name.clone()), FirNodeType::PrimOp2Expr(op), None);
        let node_id = self.graph.add_node(node);

        let op0_edge = FirEdge::new(lhs_expr, None, FirEdgeType::Operand0);
        self.graph.add_edge(lhs_id, node_id, op0_edge);

        let op1_edge = FirEdge::new(rhs_expr, None, FirEdgeType::Operand1);
        self.graph.add_edge(rhs_id, node_id, op1_edge);

        let expr = Expr::Reference(Reference::Ref(name));
        (node_id, expr)
    }

    /// Adds a new mux node to the graph.
    pub fn add_mux(
        &mut self,
        cond_id: NodeIndex,
        cond_expr: Expr,
        true_id: NodeIndex,
        true_expr: Expr,
        false_id: NodeIndex,
        false_expr: Expr,
    ) -> (NodeIndex, Expr) {
        let name = self.namespace.next();
        let node = FirNode::new(Some(name.clone()), FirNodeType::Mux, None);
        let node_id = self.graph.add_node(node);

        let cond_edge = FirEdge::new(cond_expr, None, FirEdgeType::MuxCond);
        self.graph.add_edge(cond_id, node_id, cond_edge);

        let true_edge = FirEdge::new(true_expr, None, FirEdgeType::MuxTrue);
        self.graph.add_edge(true_id, node_id, true_edge);

        let false_edge = FirEdge::new(false_expr, None, FirEdgeType::MuxFalse);
        self.graph.add_edge(false_id, node_id, false_edge);

        let expr = Expr::Reference(Reference::Ref(name));
        (node_id, expr)
    }

    /// Adds a new wire edge between two nodes in the graph.
    pub fn add_wire(
        &mut self,
        src_id: NodeIndex,
        src_expr: Expr,
        dst_id: NodeIndex,
        dst_ref: Option<Reference>,
    ) {
        let edge = FirEdge::new(src_expr, dst_ref, FirEdgeType::Wire);
        self.graph.add_edge(src_id, dst_id, edge);
    }
}

/// The top-level FIRRTL IR structure, containing all modules and hierarchy.
#[derive(Debug, Clone)]
pub struct FirIR {
    pub version: Version,
    pub name: Identifier,
    pub annos: Annotations,
    pub graphs: IndexMap<Identifier, FirGraph>,
    pub hier: Hierarchy,
}

impl FirIR {
    /// Create a new, empty FIRRTL IR.
    pub fn new(version: Version, name: Identifier, annos: Annotations) -> Self {
        Self { version, name, annos, graphs: IndexMap::new(), hier: Hierarchy::default() }
    }

    /// Export all module graphs as GraphViz PDFs to the given directory.
    pub fn export(&self, outdir: &str, pfx: &str) -> Result<(), RippleIRErr> {
        for (module, fg) in self.graphs.iter() {
            fg.export_graphviz(
                &format!("{}/{}-{}.{}.pdf", outdir, self.name.to_string(), module, pfx),
                None,
                None,
                false)?;
        }
        Ok(())
    }

    /// Adds a new module to the IR and updates the hierarchy.
    pub fn add_module(&mut self, name: Identifier, fg: FirGraph) {
        self.graphs.insert(name, fg);

        let new_hier = Hierarchy::new(self);
        self.name = new_hier.top_name().clone();
        self.hier = new_hier;
    }
}

impl DefaultGraphVizCore<FirNode, FirEdge> for FirGraph {
    fn node_indices(self: &Self) -> petgraph::graph::NodeIndices {
        self.graph.node_indices()
    }

    fn node_weight(self: &Self, id: NodeIndex) -> Option<&FirNode> {
        self.graph.node_weight(id)
    }

    fn edge_indices(self: &Self) -> petgraph::graph::EdgeIndices {
        self.graph.edge_indices()
    }

    fn edge_endpoints(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.graph.edge_endpoints(id)
    }

    fn edge_weight(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<&FirEdge> {
        self.graph.edge_weight(id)
    }
}

impl GraphViz for FirGraph {
    fn graphviz_string(
            self: &Self,
            node_attr: Option<&NodeAttributeMap>,
            edge_attr: Option<&EdgeAttributeMap>
    ) -> Result<String, std::io::Error> {
        DefaultGraphVizCore::graphviz_string(self, node_attr, edge_attr)
    }
}

#[cfg(test)]
mod test {
    use crate::common::RippleIRErr;
    use crate::ir::typetree::typetree::*;
    use crate::ir::typetree::tnode::*;
    use crate::ir::typetree::tedge::*;
    use crate::passes::runner::run_fir_passes_from_circuit;
    use rusty_firrtl::*;
    use chirrtl_parser::parse_circuit;

    #[test]
    fn io_typetree() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/GCD.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");
        let fir = run_fir_passes_from_circuit(&circuit)?;

        for (_name, fg) in fir.graphs {
            let mut io_typetree = fg.io_typetree();
            io_typetree.flip();

            let mut expect = TypeTree::default();
            let root_id = expect.graph.add_node(TypeTreeNode::new(
                    None, TypeDirection::Incoming, TypeTreeNodeType::Fields));
            expect.root = Some(root_id);

            let io_id = expect.graph.add_node(TypeTreeNode::new(
                    Some(Identifier::Name("io".to_string())),
                    TypeDirection::Outgoing,
                    TypeTreeNodeType::Fields));

            expect.graph.add_edge(root_id, io_id, TypeTreeEdge::default());

            let value1_id = expect.graph.add_node(TypeTreeNode::new(
                    Some(Identifier::Name("value1".to_string())),
                    TypeDirection::Incoming,
                    TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(16))))));

            let value2_id = expect.graph.add_node(TypeTreeNode::new(
                    Some(Identifier::Name("value2".to_string())),
                    TypeDirection::Incoming,
                    TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(16))))));

            let loadingvalues_id = expect.graph.add_node(TypeTreeNode::new(
                    Some(Identifier::Name("loadingValues".to_string())),
                    TypeDirection::Incoming,
                    TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(1))))));

            let outputgcd_id = expect.graph.add_node(TypeTreeNode::new(
                    Some(Identifier::Name("outputGCD".to_string())),
                    TypeDirection::Outgoing,
                    TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(16))))));

            let outputvalid_id = expect.graph.add_node(TypeTreeNode::new(
                    Some(Identifier::Name("outputValid".to_string())),
                    TypeDirection::Outgoing,
                    TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(1))))));

            expect.graph.add_edge(io_id, value1_id, TypeTreeEdge::default());
            expect.graph.add_edge(io_id, value2_id, TypeTreeEdge::default());
            expect.graph.add_edge(io_id, loadingvalues_id, TypeTreeEdge::default());
            expect.graph.add_edge(io_id, outputgcd_id, TypeTreeEdge::default());
            expect.graph.add_edge(io_id, outputvalid_id, TypeTreeEdge::default());

            let clock_id = expect.graph.add_node(
                TypeTreeNode::new(Some(Identifier::Name("clock".to_string())),
                TypeDirection::Incoming,
                TypeTreeNodeType::Ground(GroundType::Clock)));

            let reset_id = expect.graph.add_node(
                TypeTreeNode::new(Some(Identifier::Name("reset".to_string())),
                TypeDirection::Incoming,
                TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(1))))));

            expect.graph.add_edge(root_id, clock_id, TypeTreeEdge::default());
            expect.graph.add_edge(root_id, reset_id, TypeTreeEdge::default());
            expect.assign_unique_id();

            assert!(io_typetree.view().unwrap().eq(&expect.view().unwrap()));
        }

        Ok(())
    }
}
