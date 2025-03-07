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
use crate::parser::Int;
use crate::common::graphviz::GraphViz;
use petgraph::graph::{Graph, NodeIndex};
use std::fmt::Display;

#[derive(Debug, Default, Clone, PartialEq, Hash)]
pub enum NodeType {
    #[default]
    Invalid,

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
    Inst(Identifier, Identifier),
// Node(Identifier, Expr),
    // TODO: deal with printfs

    // Port
    Input(Identifier, Type),
    Output(Identifier, Type),
    Phi,
}

impl Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let original = format!("{:?}", self);
        let clean_for_dot = original.replace('"', "");
        write!(f, "{}", clean_for_dot)
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum EdgeType {
    Default,
    Wire(Identifier, Type)
}

impl Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub type GraphIR = Graph<NodeType, EdgeType>;

#[derive(Debug, Clone)]
pub struct RippleIR {
    pub graph: GraphIR
}

impl RippleIR {
    pub fn new() -> Self {
        Self { graph: GraphIR::new() }
    }
}

impl GraphViz<NodeType, EdgeType> for RippleIR {
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



