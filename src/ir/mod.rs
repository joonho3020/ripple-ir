// Nodes
// - Primops:
//   - PrimOp2Expr
//   - PrimOp1Expr
//   - PrimOp1Expr1Int
//   - PrimOp1Expr2Int
// - Stmt:
//   - Reg
//   - RegReset
//   - ChirrtlMemory
//   - Inst
//   - Node
//   - Printf
//   - Assert
// - Port
// - Expr:
//  - UIntNoInit
//  - UIntInit
//  - SIntInit
//  - SInt
//
// Edges
//
// - Stmt
//   - connect (x, y)
//   - node (x, y)

use crate::parser::ast::*;
use crate::parser::whentree::Condition;
use crate::parser::Int;
use crate::common::graphviz::GraphViz;
use indexmap::IndexMap;
use petgraph::graph::{Graph, NodeIndex};
use std::fmt::Display;

#[derive(Debug, Default, Clone, PartialEq, Hash)]
pub enum NodeType {
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
    Wire(Identifier, Type),
    Reg(Identifier, Type, Expr),
    RegReset(Identifier, Type, Expr, Expr, Expr),
    SMem(Identifier, Type, Option<ChirrtlMemoryReadUnderWrite>),
    CMem(Identifier, Type),
    WriteMemPort(Identifier),
    ReadMemPort(Identifier),
    InferMemPort(Identifier),
    Inst(Identifier, Identifier),

    // TODO: deal with printfs

    // Port
    Input(Identifier, Type),
    Output(Identifier, Type),
    Phi(Identifier),
}

impl Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let original = format!("{:?}", self);
        let clean_for_dot = original.replace('"', "");
        write!(f, "{}", clean_for_dot)
    }
}

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

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum EdgeType {
    /// Reference <- Expr
    Wire(Reference, Expr),

    Operand0(Expr),
    Operand1(Expr),

    MuxCond,
    MuxTrue(Expr),
    MuxFalse(Expr),

    /// Clock edge
    Clock(Reference),

    /// Reset edge
    Reset(Reference),

    /// Represents don't cares
    DontCare(Reference),

    /// Edge going into the phi node
    PhiInput(PhiPriority, Condition, Reference, Expr),

    /// Selection conditions going into the phi node
    PhiSel(Expr),

    /// Edge comming out from phi node
    PhiOut,

    /// Connects the memory to its ports
    MemPortEdge,

    /// Connects the address signal to the memory port node
    MemPortAddr(Expr),
}

impl Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let original = format!("{:?}", self);
        let clean_for_dot = original.replace('"', "");
        write!(f, "{}", clean_for_dot)
    }
}

pub type IRGraph = Graph<NodeType, EdgeType>;

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

impl GraphViz<NodeType, EdgeType> for RippleGraph {
    fn node_indices(self: &Self) -> petgraph::graph::NodeIndices {
        self.graph.node_indices()
    }

    fn node_weight(self: &Self, id: NodeIndex) -> Option<&NodeType> {
        self.graph.node_weight(id)
    }

    fn edge_indices(self: &Self) -> petgraph::graph::EdgeIndices {
        self.graph.edge_indices()
    }

    fn edge_endpoints(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.graph.edge_endpoints(id)
    }

    fn edge_weight(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<&EdgeType> {
        self.graph.edge_weight(id)
    }
}
