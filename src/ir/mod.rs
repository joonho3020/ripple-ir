pub mod typetree;
pub mod whentree;
pub mod firir;

use chirrtl_parser::ast::*;
use derivative::Derivative;
use std::fmt::Display;
use indexmap::{IndexMap, IndexSet};
use petgraph::graph::{EdgeIndex, EdgeReference, Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use crate::common::graphviz::GraphViz;
use crate::ir::whentree::Condition;

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pin(Reference);

#[derive(Debug, Clone)]
pub struct PinMap {
    pub map: Graph<Pin, ()>,
    pub ipins: IndexMap<Pin, NodeIndex>,
    pub opins: IndexMap<Pin, NodeIndex>,
}

impl PinMap {
    pub fn connected_pins(&self, pin: &Pin, dir: petgraph::Direction) -> Vec<&Pin> {
        let pin_id = self.ipins.get(pin).unwrap();
        let conn_pins = self.map.neighbors_directed(*pin_id, dir);
        conn_pins.map(|id| self.map.node_weight(id).unwrap()).collect()
    }
}

#[derive(Derivative, Clone)]
#[derivative(Debug)]
pub struct RippleNode {
    pub name: Option<Identifier>,

    pub tpe: RippleNodeType,

    #[derivative(Debug="ignore")]
    pub pinmap: PinMap,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RippleEdge {
    pub src: Pin,
    pub dst: Pin,
    pub et: RippleEdgeType
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

type IRGraph = Graph<RippleNode, RippleEdge>;

#[derive(Debug, Clone)]
pub struct RippleGraph {
    pub graph: IRGraph,
}

impl RippleGraph {
    pub fn new() -> Self {
        Self { graph: IRGraph::new() }
    }

    /// Given an `id` of the graph node, the `pin` that we are looking at, and the direction,
    /// returns a reference of edges that are connected to this `pin`
    pub fn neighbor_pins_directed(&self, id: NodeIndex, pin: &Pin, dir: petgraph::Direction) -> Vec<EdgeReference<RippleEdge>> {
        let node = self.graph.node_weight(id).unwrap();
        let opins = node.pinmap.connected_pins(pin, dir);
        let opins_set: IndexSet<&Pin> = IndexSet::from_iter(opins);

        self.graph.edges_directed(id, dir).filter(|x| {
            let e = self.graph.edge_weight(x.id()).unwrap();
            opins_set.contains(&e.src)
        }).collect()
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
