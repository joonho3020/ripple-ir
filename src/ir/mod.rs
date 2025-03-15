pub mod typetree;
pub mod whentree;

use chirrtl_parser::ast::*;
use derivative::Derivative;
use std::fmt::Display;
use indexmap::IndexMap;
use petgraph::graph::{Graph, NodeIndex};
use crate::common::graphviz::GraphViz;
use crate::ir::typetree::TypeTree;
use crate::ir::whentree::Condition;

#[derive(Debug, Default, Clone, PartialEq, Hash)]
pub struct PhiPriority {
    /// Priority between blocks
    /// - Smaller number means higher priority
    pub block: u32,

    /// Priority between statements within the same block
    /// - Smaller number means higher priority
    pub stmt: u32,
}

impl PhiPriority {
    pub fn new(block: u32, stmt: u32) -> Self {
        Self { block, stmt }
    }
}


// TODO: fill in the ttree field after graph construction has been finished
#[derive(Derivative, Clone)]
#[derivative(Debug)]
pub struct RippleNode {
    pub name: Option<Identifier>,

    pub nt: RippleNodeType,

    #[derivative(Debug="ignore")]
    pub ttree: Option<TypeTree>,
}

impl RippleNode {
    pub fn new(name: Option<Identifier>, nt: RippleNodeType, ttree: Option<TypeTree>) -> Self {
        Self { name, nt, ttree }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Hash)]
pub enum RippleNodeType {
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
    WriteMemPort,
    ReadMemPort,
    InferMemPort,
    Inst(Identifier),

    // Port
    Input,
    Output,
    Phi,
}

impl Display for RippleNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let original = format!("{:?}", self);
        let clean_for_dot = original.replace('"', "");
        write!(f, "{}", clean_for_dot)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RippleEdge {
    pub src: Expr,
    pub dst: Option<Reference>,
    pub et: RippleEdgeType
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum RippleEdgeType {
    Wire,

    Operand0,
    Operand1,

    MuxCond,
    MuxTrue,
    MuxFalse,

    Clock,
    Reset,
    DontCare,

    PhiInput(PhiPriority, Condition),
    PhiSel,
    PhiOut,

    MemPortEdge,
    MemPortAddr,

    ArrayAddr,
}

impl Display for RippleEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let original = format!("{:?}", self);
        let clean_for_dot = original.replace('"', "");
        write!(f, "{}", clean_for_dot)
    }
}

impl RippleEdge {
    pub fn new(src: Expr, dst: Option<Reference>, et: RippleEdgeType) -> Self {
        Self { src, et, dst }
    }
}

type IRGraph = Graph<RippleNode, RippleEdge>;

#[derive(Debug, Clone)]
pub struct RippleGraph {
    pub graph: IRGraph,
}

impl RippleGraph {
    pub fn new() -> Self {
        Self { graph: IRGraph::new() }
    }
}

#[derive(Debug, Default, Clone)]
pub struct RippleIR {
    pub graphs: IndexMap<Identifier, RippleGraph>
}

impl RippleIR {
    pub fn new() -> Self {
        Self { graphs: IndexMap::new() }
    }
}

impl GraphViz<RippleNode, RippleEdge> for RippleGraph {
    fn node_indices(self: &Self) -> petgraph::graph::NodeIndices {
        self.graph.node_indices()
    }

    fn node_weight(self: &Self, id: NodeIndex) -> Option<&RippleNode> {
        self.graph.node_weight(id)
    }

    fn edge_indices(self: &Self) -> petgraph::graph::EdgeIndices {
        self.graph.edge_indices()
    }

    fn edge_endpoints(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.graph.edge_endpoints(id)
    }

    fn edge_weight(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<&RippleEdge> {
        self.graph.edge_weight(id)
    }
}
