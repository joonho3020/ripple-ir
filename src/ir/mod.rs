pub mod typetree;
pub mod whentree;
pub mod firir;

use chirrtl_parser::ast::*;
use derivative::Derivative;
use std::fmt::Display;
use std::hash::Hash;
use indexmap::IndexMap;
use petgraph::graph::{Graph, NodeIndex, EdgeIndex};
use crate::common::graphviz::GraphViz;
use crate::ir::whentree::Condition;
use crate::ir::typetree::*;
use crate::ir::firir::*;

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

#[derive(Derivative, Clone)]
#[derivative(Debug)]
pub struct RippleNode {
    pub name: Option<Identifier>,
    pub tpe: RippleNodeType,
    pub tg: GroundType
}

impl RippleNode {
    pub fn new(name: Option<Identifier>, tpe: RippleNodeType, tg: GroundType) -> Self {
        Self { name, tpe, tg }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub enum RippleNodeType {
    #[default]
    Invalid,

    DontCare,

    UIntLiteral(Int),
    SIntLiteral(Int),

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

impl From<&FirNodeType> for RippleNodeType {
    fn from(value: &FirNodeType) -> Self {
        match value {
            FirNodeType::Invalid => Self::Invalid,
            FirNodeType::DontCare => Self::DontCare,
            FirNodeType::UIntLiteral(_, x) => Self::UIntLiteral(x.clone()),
            FirNodeType::SIntLiteral(_, x) => Self::SIntLiteral(x.clone()),
            FirNodeType::Mux => Self::Mux,
            FirNodeType::PrimOp2Expr(op) => Self::PrimOp2Expr(op.clone()),
            FirNodeType::PrimOp1Expr(op) => Self::PrimOp1Expr(op.clone()),
            FirNodeType::PrimOp1Expr1Int(op, a) => Self::PrimOp1Expr1Int(op.clone(), a.clone()),
            FirNodeType::PrimOp1Expr2Int(op, a, b) => Self::PrimOp1Expr2Int(op.clone(), a.clone(), b.clone()),
            FirNodeType::Wire => Self::Wire,
            FirNodeType::Reg => Self::Reg,
            FirNodeType::RegReset => Self::RegReset,
            FirNodeType::SMem(x) => Self::SMem(x.clone()),
            FirNodeType::CMem => Self::CMem,
            FirNodeType::WriteMemPort => Self::WriteMemPort,
            FirNodeType::ReadMemPort => Self::ReadMemPort,
            FirNodeType::InferMemPort => Self::InferMemPort,
            FirNodeType::Inst(x) => Self::Inst(x.clone()),
            FirNodeType::Input => Self::Input,
            FirNodeType::Output => Self::Output,
            FirNodeType::Phi => Self::Phi,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RippleEdge {
    pub width: Option<Width>,
    pub et: RippleEdgeType
}

impl RippleEdge {
    pub fn new(width: Option<Width>, et: RippleEdgeType) -> Self {
        Self { width, et }
    }
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

impl From<&FirEdgeType> for RippleEdgeType {
    fn from(value: &FirEdgeType) -> Self {
        match value {
            FirEdgeType::Wire => RippleEdgeType::Wire,
            FirEdgeType::Operand0 => RippleEdgeType::Operand0,
            FirEdgeType::Operand1 => RippleEdgeType::Operand1,
            FirEdgeType::MuxCond => RippleEdgeType::MuxCond,
            FirEdgeType::MuxTrue => RippleEdgeType::MuxTrue,
            FirEdgeType::MuxFalse => RippleEdgeType::MuxFalse,
            FirEdgeType::Clock => RippleEdgeType::Clock,
            FirEdgeType::Reset => RippleEdgeType::Reset,
            FirEdgeType::DontCare => RippleEdgeType::DontCare,
            FirEdgeType::PhiInput(prior, cond) => RippleEdgeType::PhiInput(prior.clone(), cond.clone()),
            FirEdgeType::PhiSel => RippleEdgeType::PhiSel,
            FirEdgeType::PhiOut => RippleEdgeType::PhiOut,
            FirEdgeType::MemPortEdge => RippleEdgeType::MemPortEdge,
            FirEdgeType::MemPortAddr => RippleEdgeType::MemPortAddr,
            FirEdgeType::ArrayAddr => RippleEdgeType::ArrayAddr
        }
    }
}

type IRGraph = Graph<RippleNode, RippleEdge>;

pub type TreeIdx = u32;

#[derive(Debug, Clone)]
pub struct TypeTreeIdx {
    leaf_id: NodeIndex,
    tree_id: TreeIdx,
}

impl TypeTreeIdx {
    pub fn new(leaf_id: NodeIndex, tree_id: TreeIdx) -> Self {
        Self { leaf_id, tree_id }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RootTypeTreeKey {
    pub name: Identifier,
    pub nt: RippleNodeType,
}

impl RootTypeTreeKey {
    pub fn new(name: Identifier, nt: RippleNodeType) -> Self {
        Self { name, nt }
    }
}

#[derive(Debug, Clone)]
pub struct RippleGraph {
    pub graph: IRGraph,
    pub ttree_idx_map: IndexMap<NodeIndex, TypeTreeIdx>,
    pub root_ref_ttree_idx_map: IndexMap<RootTypeTreeKey, TreeIdx>,
    pub ttrees: Vec<TypeTree>,
}

impl RippleGraph {
    pub fn new() -> Self {
        Self {
            graph: IRGraph::new(),
            ttree_idx_map: IndexMap::new(),
            root_ref_ttree_idx_map: IndexMap::new(),
            ttrees: vec![],
        }
    }

    pub fn add_node(&mut self, node: RippleNode) -> NodeIndex {
        self.graph.add_node(node)
    }

    pub fn add_aggregate_node(
        &mut self,
        name: Identifier,
        ttree: &TypeTree,
        nt: RippleNodeType
    ) -> RootTypeTreeKey {
        let mut my_ttree = ttree.clone();
        let ttree_id = self.ttrees.len() as u32;
        let leaves = my_ttree.all_leaves();

        // Add all the leaf nodes
        for leaf_id in leaves {
            let leaf = my_ttree.graph.node_weight(leaf_id).unwrap();
            let tg = match &leaf.tpe {
                TypeTreeNodeType::Ground(x) => x,
                _ => panic!("Type tree leaves doesn't have Ground, got {:?}", leaf)
            };

            let leaf_name = my_ttree.node_name(&name, leaf_id);

            // Insert new graph node
            let rgnode = RippleNode::new(Some(leaf_name), nt.clone(), tg.clone());
            let rg_id = self.add_node(rgnode);

            // Add this node to the ttree_idx_map
            self.ttree_idx_map.insert(rg_id, TypeTreeIdx::new(leaf_id, ttree_id));

            // Update ttree to point to this node
            my_ttree.graph.node_weight_mut(leaf_id).unwrap().id = Some(rg_id);
        }

        // Add the type tree
        self.ttrees.push(my_ttree);

        let root_key = RootTypeTreeKey::new(name, nt);
        self.root_ref_ttree_idx_map.insert(root_key.clone(), ttree_id);
        return root_key;
    }

    pub fn add_edge(&mut self, src: NodeIndex, dst: NodeIndex, edge: RippleEdge) -> EdgeIndex {
        self.graph.add_edge(src, dst, edge)
    }

    pub fn add_aggregate_edge(
        &mut self,
        src_key: &RootTypeTreeKey,
        src_ref: &Reference,
        dst_key: &RootTypeTreeKey,
        dst_ref: &Reference,
        et: RippleEdgeType
    ) {
        let src_ttree_id = self.root_ref_ttree_idx_map.get(src_key).expect("to exist");
        let src_ttree = self.ttrees.get(*src_ttree_id as usize).expect("to exist");
        let src_leaves = src_ttree.subtree_leaves_with_path(src_ref);


        let dst_ttree_id = self.root_ref_ttree_idx_map.get(dst_key).expect("to exist");
        let dst_ttree = self.ttrees.get(*dst_ttree_id as usize).expect("to exist");
        let dst_leaves = dst_ttree.subtree_leaves_with_path(dst_ref);

        println!("src_ref {:?} dst_ref {:?}", src_ref, dst_ref);
        println!("{:?} src_leaves: {:?}", src_key, src_leaves);
        println!("{:?} dst_leaves: {:?}", dst_key, dst_leaves);

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        for (src_path_key, src_ttree_leaf_id) in src_leaves {
            let src_ttree_leaf = src_ttree.graph.node_weight(src_ttree_leaf_id).unwrap();
            if dst_leaves.contains_key(&src_path_key) {
                let dst_ttree_leaf_id = dst_leaves.get(&src_path_key).unwrap();
                let dst_ttree_leaf = dst_ttree.graph.node_weight(*dst_ttree_leaf_id).unwrap();
                if src_ttree_leaf.dir == TypeDirection::Outgoing {
                    edges.push((
                        src_ttree_leaf.id.unwrap(),
                        dst_ttree_leaf.id.unwrap(),
                        RippleEdge::new(None, et.clone())));
                } else {
                    edges.push((
                        dst_ttree_leaf.id.unwrap(),
                        src_ttree_leaf.id.unwrap(),
                        RippleEdge::new(None, et.clone())));
                }
            } else {
                panic!("Not connected src_ref {:?} src_key {:?} dst_ref {:?} dst_key {:?}",
                    src_key, src_ref, dst_ref, dst_key);
            }
        }

        // TODO: Test for partial connections?

        for edge in edges {
            self.add_edge(edge.0, edge.1, edge.2);
        }
    }
}

#[derive(Debug, Clone)]
pub struct RippleIR {
    pub name: Identifier,
    pub graphs: IndexMap<Identifier, RippleGraph>
}

impl RippleIR {
    pub fn new(name: Identifier) -> Self {
        Self { name, graphs: IndexMap::new() }
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
