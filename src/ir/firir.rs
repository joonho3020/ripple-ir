use chirrtl_parser::ast::*;
use derivative::Derivative;
use std::fmt::Display;
use indexmap::IndexMap;
use petgraph::graph::{Graph, NodeIndex};
use crate::common::graphviz::GraphViz;
use crate::ir::typetree::TypeTree;
use crate::ir::whentree::Condition;
use crate::ir::PhiPriority;

#[derive(Derivative, Clone)]
#[derivative(Debug)]
pub struct FirNode {
    pub name: Option<Identifier>,

    pub nt: FirNodeType,

    #[derivative(Debug="ignore")]
    pub ttree: Option<TypeTree>,
}

impl FirNode {
    pub fn new(name: Option<Identifier>, nt: FirNodeType, ttree: Option<TypeTree>) -> Self {
        Self { name, nt, ttree }
    }
}

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
    WriteMemPort,
    ReadMemPort,
    InferMemPort,
    Inst(Identifier),

    // Port
    Input,
    Output,
    Phi,
}

impl Display for FirNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let original = format!("{:?}", self);
        let clean_for_dot = original.replace('"', "");
        write!(f, "{}", clean_for_dot)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FirEdge {
    pub src: Expr,
    pub dst: Option<Reference>,
    pub et: FirEdgeType
}

#[derive(Derivative, Clone, PartialEq, Hash)]
#[derivative(Debug)]
pub enum FirEdgeType {
    Wire,

    Operand0,
    Operand1,

    MuxCond,
    MuxTrue,
    MuxFalse,

    Clock,
    Reset,
    DontCare,

    PhiInput(PhiPriority, #[derivative(Debug="ignore")] Condition),
    PhiSel,
    PhiOut,

    MemPortEdge,
    MemPortAddr,

    ArrayAddr,
}

impl Display for FirEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let original = format!("{:?}", self);
        let clean_for_dot = original.replace('"', "");
        write!(f, "{}", clean_for_dot)
    }
}

impl FirEdge {
    pub fn new(src: Expr, dst: Option<Reference>, et: FirEdgeType) -> Self {
        Self { src, et, dst }
    }
}

type IRGraph = Graph<FirNode, FirEdge>;

#[derive(Debug, Clone)]
pub struct FirGraph {
    pub graph: IRGraph,
}

impl FirGraph {
    pub fn new() -> Self {
        Self { graph: IRGraph::new() }
    }
}

#[derive(Debug, Default, Clone)]
pub struct FirIR {
    pub graphs: IndexMap<Identifier, FirGraph>
}

impl FirIR {
    pub fn new() -> Self {
        Self { graphs: IndexMap::new() }
    }
}

impl GraphViz<FirNode, FirEdge> for FirGraph {
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
