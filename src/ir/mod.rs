pub mod typetree;
pub mod whentree;
pub mod firir;

use chirrtl_parser::ast::*;
use derivative::Derivative;
use std::fmt::Display;
use indexmap::IndexMap;
use petgraph::graph::{Graph, NodeIndex, EdgeIndex};
use crate::common::graphviz::GraphViz;
use crate::ir::whentree::Condition;
use crate::ir::typetree::*;

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

#[derive(Debug, Default, Clone, PartialEq, Hash)]
pub enum RippleNodeType {
    #[default]
    Invalid,

    DontCare,

    UIntLiteral(Int),
    SIntLiteral(Int),

    Mux,
    Op,

    // Stmt
    Wire,
    Reg,
    RegReset,
    SMem,
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

#[derive(Debug, Clone)]
pub struct RippleGraph {
    pub graph: IRGraph,
    pub ttree_idx_map: IndexMap<NodeIndex, TypeTreeIdx>,
    pub root_ref_ttree_idx_map: IndexMap<Identifier, TreeIdx>,
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

    pub fn add_aggregate_node(&mut self, name: Identifier, tpe: &Type, nt: RippleNodeType) {
        let mut ttree = TypeTree::build_from_type(tpe, Direction::Output);
        let ttree_id = self.ttrees.len() as u32;
        let leaves = ttree.all_leaves();

        // Add all the leaf nodes
        for leaf_id in leaves {
            let leaf = ttree.graph.node_weight(leaf_id).unwrap();
            let tg = match &leaf.tpe {
                TypeTreeNodeType::Ground(x) => x,
                _ => panic!("Type tree leaves doesn't have Ground, got {:?}", leaf)
            };

            // Insert new graph node
            let rgnode = RippleNode::new(Some(leaf.name.clone().unwrap()), nt.clone(), tg.clone());
            let rg_id = self.add_node(rgnode);

            // Add this node to the ttree_idx_map
            self.ttree_idx_map.insert(rg_id, TypeTreeIdx::new(leaf_id, ttree_id));

            // Update ttree to point to this node
            ttree.graph.node_weight_mut(leaf_id).unwrap().id = Some(rg_id);
        }

        // Add the type tree
        self.ttrees.push(ttree);
        self.root_ref_ttree_idx_map.insert(name, ttree_id);
    }

    pub fn add_edge(&mut self, src: NodeIndex, dst: NodeIndex, edge: RippleEdge) -> EdgeIndex {
        self.graph.add_edge(src, dst, edge)
    }

    pub fn add_aggregate_edge(
        &mut self,
        src_name: Identifier,
        src_ref: &Reference,
        dst_name: Identifier,
        dst_ref: &Reference,
        et: RippleEdgeType
    ) {
        let src_ttree_id = self.root_ref_ttree_idx_map.get(&src_name).expect("to exist");
        let src_ttree = self.ttrees.get(*src_ttree_id as usize).expect("to exist");
        let src_leaves = src_ttree.subtree_leaves_with_path(src_ref);

        let dst_ttree_id = self.root_ref_ttree_idx_map.get(&dst_name).expect("to exist");
        let dst_ttree = self.ttrees.get(*dst_ttree_id as usize).expect("to exist");
        let dst_leaves = dst_ttree.subtree_leaves_with_path(dst_ref);

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        for (src_path_key, src_ttree_leaf_id) in src_leaves {
            let src_ttree_leaf = src_ttree.graph.node_weight(src_ttree_leaf_id).unwrap();
            if dst_leaves.contains_key(&src_path_key) {
                let dst_ttree_leaf_id = dst_leaves.get(&src_path_key).unwrap();
                let dst_ttree_leaf = dst_ttree.graph.node_weight(*dst_ttree_leaf_id).unwrap();
                if src_ttree_leaf.dir == Direction::Output {
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
            }
        }

        // TODO: Test for partial connections?

        for edge in edges {
            self.add_edge(edge.0, edge.1, edge.2);
        }
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
