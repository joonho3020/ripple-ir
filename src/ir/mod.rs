pub mod typetree;
pub mod whentree;
pub mod firir;
pub mod hierarchy;

use chirrtl_parser::ast::*;
use derivative::Derivative;
use hierarchy::Hierarchy;
use petgraph::visit::EdgeRef;
use petgraph::Direction::Incoming;
use petgraph::Direction::Outgoing;
use std::hash::Hash;
use indexmap::{IndexMap, IndexSet};
use petgraph::graph::{Graph, NodeIndex, EdgeIndex};
use fixedbitset::FixedBitSet;
use crate::common::graphviz::*;
use crate::ir::whentree::Condition;
use crate::ir::typetree::*;
use crate::ir::firir::*;
use crate::impl_clean_display;

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

#[derive(Derivative, Clone, PartialEq, Eq, Hash)]
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

impl_clean_display!(RippleNodeType);
impl_clean_display!(RippleNode);

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

impl_clean_display!(RippleEdge);

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggEdgeIdentifier {
    pub dst_id: AggNodeIndex,
    pub et: RippleEdgeType,
}

impl AggEdgeIdentifier {
    pub fn new(dst_id: AggNodeIndex, et: RippleEdgeType) -> Self {
        Self { dst_id, et }
    }
}

#[derive(Debug, Clone)]
pub struct AggVisMap {
    visited: FixedBitSet
}

impl AggVisMap {
    pub fn new(num_bits: u32) -> Self {
        Self { visited: FixedBitSet::with_capacity(num_bits as usize) }
    }

    pub fn is_visited(&self, id: AggNodeIndex) -> bool {
        self.visited.contains(id.into())
    }

    pub fn visit(&mut self, id: AggNodeIndex) {
        self.visited.set(id.into(), true);
    }

    pub fn has_unvisited(&self) -> bool {
        self.visited.count_zeroes(..) > 0
    }

    pub fn unvisited_ids(&self) -> Vec<AggNodeIndex> {
        self.visited.zeroes().map(|x| x.into()).collect()
    }
}

type IRGraph = Graph<RippleNode, RippleEdge>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct AggNodeIndex(u32);

impl AggNodeIndex {
    pub fn to_usize(&self) -> usize {
        self.0 as usize
    }
}

impl Into<u32> for AggNodeIndex {
    fn into(self) -> u32 {
        self.0 as u32
    }
}

impl From<u32> for AggNodeIndex {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<usize> for AggNodeIndex {
    fn from(value: usize) -> Self {
        Self(value as u32)
    }
}

impl From<AggNodeIndex> for usize {
    fn from(value: AggNodeIndex) -> Self {
        value.0 as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggNodeLeafIndex {
    /// Aggregate node id
    pub agg_id: AggNodeIndex,

    /// Vec of leaf nodes that this graph node corresponds to
    /// - Usually length is 1
    /// - For memory or array types, a single graph node can correspond to multiple leaf nodes
    /// in a TypeTree
    pub leaf_ids: Vec<NodeIndex>,
}

impl AggNodeLeafIndex {
    pub fn new(agg_id: AggNodeIndex, leaf_ids: Vec<NodeIndex>) -> Self {
        Self { agg_id, leaf_ids }
    }
}

/// Can be used as a key to identify a `TypeTree` in `RippleGraph`
/// Represents a unique aggregate node in the IR
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggNodeIdentifier {
    /// Identifier of the reference root
    pub name: Identifier,

    /// Need to identify the type of the node as multiple nodes can
    /// use the same Identifier but have different `RippleNodeType`.
    /// E.g., Phi and Reg nodes
    pub nt: RippleNodeType,
}

impl AggNodeIdentifier {
    pub fn new(name: Identifier, nt: RippleNodeType) -> Self {
        Self { name, nt }
    }
}

#[derive(Debug, Clone)]
pub struct RippleGraph {
    /// Graph of this IR
    pub graph: IRGraph,

    /// Maps each node in `graph` to `AggNodeIndex`
    pub id_to_agg_leaf: IndexMap<NodeIndex, AggNodeLeafIndex>,

    /// Maps a key that represents a unique aggregate node `AggNodeIdentifier` to the `TypeTree`
    pub agg_identifier_to_id: IndexMap<AggNodeIdentifier, AggNodeIndex>,

    /// Maps a `TypeTree` to a `AggNodeIdentifier`
    pub agg_id_to_identifier: IndexMap<AggNodeIndex, AggNodeIdentifier>,

    /// `TypeTree`. Each represents an aggregate node (or it can be a single node
    /// for nodes with `GroundType`s)
    pub ttrees: Vec<TypeTree>,
}

impl RippleGraph {
    pub fn new() -> Self {
        Self {
            graph: IRGraph::new(),
            id_to_agg_leaf: IndexMap::new(),
            agg_identifier_to_id: IndexMap::new(),
            agg_id_to_identifier: IndexMap::new(),
            ttrees: vec![],
        }
    }

    /// Removes a single low-level node from the IR
    pub fn remove_node(&mut self, id: NodeIndex) {
        let last_id = NodeIndex::new(self.graph.node_count() - 1);
        self.graph.remove_node(id);

        // Remove id from id_to_agg_leaf
        let agg_leaf = self.id_to_agg_leaf.swap_remove(&id).unwrap();

        // Unmap the id from the typetree leaf
        let ttree = self.ttrees.get_mut(agg_leaf.agg_id.to_usize()).unwrap();
        for leaf_id in agg_leaf.leaf_ids {
            ttree.graph.node_weight_mut(leaf_id).unwrap().id = None;
        }

        if last_id != id {
            // Remove the last id from id_to_agg_leaf
            let last_agg_leaf = self.id_to_agg_leaf.swap_remove(&last_id).unwrap();

            // Map the typetree leaf that had last_id to id
            let last_ttree = self.ttrees.get_mut(last_agg_leaf.agg_id.to_usize()).unwrap();
            for leaf_id in last_agg_leaf.leaf_ids.iter() {
                last_ttree.graph.node_weight_mut(*leaf_id).unwrap().id = Some(id);
            }

            // Map id to last_agg_leaf
            self.id_to_agg_leaf.insert(id, last_agg_leaf);
        }
    }

    fn add_node(&mut self, node: RippleNode) -> NodeIndex {
        self.graph.add_node(node)
    }

    pub fn add_aggregate_node(
        &mut self,
        name: Identifier,
        ttree: &TypeTree,
        nt: RippleNodeType
    ) -> AggNodeIdentifier {
        let mut my_ttree = ttree.clone();
        let ttree_id = self.ttrees.len() as u32;
        let leaves = my_ttree.all_leaves();

        // Add all the leaf nodes
        for leaf_id in leaves {
            let leaf = my_ttree.graph.node_weight(leaf_id).unwrap();
            let tg = match &leaf.tpe {
                TypeTreeNodeType::Ground(x) => x,
                _ => {
                    // Empty aggregate type, add a invalid ground type
                    &GroundType::Invalid
                }
            };

            let leaf_name = my_ttree.node_name(&name, leaf_id);

            // Insert new graph node
            let rgnode = RippleNode::new(Some(leaf_name), nt.clone(), tg.clone());
            let rg_id = self.add_node(rgnode);

            // Add this node to the id_to_agg_leaf
            self.id_to_agg_leaf.insert(rg_id, AggNodeLeafIndex::new(AggNodeIndex::from(ttree_id), vec![leaf_id]));

            // Update ttree to point to this node
            my_ttree.graph.node_weight_mut(leaf_id).unwrap().id = Some(rg_id);
        }

        // Add the type tree
        self.ttrees.push(my_ttree);

        let agg_identity = AggNodeIdentifier::new(name, nt);
        self.agg_identifier_to_id.insert(agg_identity.clone(), ttree_id.into());
        self.agg_id_to_identifier.insert(ttree_id.into(), agg_identity.clone());
        return agg_identity;
    }

    pub fn add_edge(&mut self, src: NodeIndex, dst: NodeIndex, edge: RippleEdge) -> EdgeIndex {
        self.graph.add_edge(src, dst, edge)
    }

    pub fn add_single_edge(
        &mut self,
        src_identity: &AggNodeIdentifier,
        src_ref: &Reference,
        dst_identity: &AggNodeIdentifier,
        dst_ref: &Reference,
        et: RippleEdgeType
    ) {
        let src_ttree_id = self.agg_identifier_to_id.get(src_identity).expect("to exist");
        let src_ttree = self.ttrees.get(src_ttree_id.to_usize()).expect("to exist");
        let src_leaves = src_ttree.subtree_leaves_with_path(src_ref);

        let dst_ttree_id = self.agg_identifier_to_id.get(dst_identity).expect("to exist");
        let dst_ttree = self.ttrees.get(dst_ttree_id.to_usize()).expect("to exist");
        let dst_leaves = dst_ttree.subtree_leaves_with_path(dst_ref);

        assert!(src_leaves.len() == 1, "add_single_edge got multiple src_leaves {:?}", src_leaves);
        assert!(dst_leaves.len() > 0, "add_single_edge got zero destinations");

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        let (_src_path_identity, src_ttree_leaf_id) = src_leaves.first().unwrap();
        let src_ttree_leaf = src_ttree.graph.node_weight(*src_ttree_leaf_id).unwrap();

        for (_dst_path_identity, dst_ttree_leaf_id) in dst_leaves {
            let dst_ttree_leaf = dst_ttree.graph.node_weight(dst_ttree_leaf_id).unwrap();

            if dst_ttree_leaf.id.is_none() ||
               src_ttree_leaf.id.is_none()
            {
                src_ttree.print_tree();
                dst_ttree.print_tree();
                println!("{:?} {:?} {:?} {:?}", src_identity, src_ref, dst_identity, dst_ref);
                println!("dst_ttree_leaf {:?}", dst_ttree_leaf);
            }

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
        }

        for edge in edges {
            self.add_edge(edge.0, edge.1, edge.2);
        }
    }

    pub fn add_aggregate_edge(
        &mut self,
        src_identity: &AggNodeIdentifier,
        src_ref: &Reference,
        dst_identity: &AggNodeIdentifier,
        dst_ref: &Reference,
        et: RippleEdgeType
    ) {
        // Get leaves of the src aggregate node
        let src_ttree_id = self.agg_identifier_to_id.get(src_identity).expect("to exist");
        let src_ttree = self.ttrees.get(src_ttree_id.to_usize()).expect("to exist");
        let src_leaves = src_ttree.subtree_leaves_with_path(src_ref);

        // Get leaves of the dst aggregate node
        let dst_ttree_id = self.agg_identifier_to_id.get(dst_identity).expect("to exist");
        let dst_ttree = self.ttrees.get(dst_ttree_id.to_usize()).expect("to exist");
        let dst_leaves = dst_ttree.subtree_leaves_with_path(dst_ref);

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        for (src_path_identity, src_ttree_leaf_id) in src_leaves.iter() {
            let src_ttree_leaf = src_ttree.graph.node_weight(*src_ttree_leaf_id).unwrap();

            // If there is a matching path in the dst aggregate node, add an edge
            if dst_leaves.contains_key(src_path_identity) {
                let dst_ttree_leaf_id = dst_leaves.get(src_path_identity).unwrap();
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
                println!("EdgeType {:?}", et);
                panic!("Not connected src_ref {:?}\nsrc_identity {:?}\nsrc_leaves {:?}\ndst_ref {:?}\ndst_identity {:?}\ndst_leaves {:?}",
                    src_ref, src_identity, src_leaves, dst_ref, dst_identity, dst_leaves);
            }
        }

        for edge in edges {
            self.add_edge(edge.0, edge.1, edge.2);
        }
    }

    pub fn add_aggregate_mem_edge(
        &mut self,
        src_identity: &AggNodeIdentifier,
        src_ref: &Reference,
        dst_identity: &AggNodeIdentifier,
        dst_ref: &Reference,
        et: RippleEdgeType
    ) {
        // Get leaves of the src aggregate node
        let src_ttree_id = self.agg_identifier_to_id.get(src_identity).expect("to exist");
        let src_ttree = self.ttrees.get(src_ttree_id.to_usize()).expect("to exist");
        let src_ttree_array_entry = src_ttree.subtree_array_element();
        let src_leaves = src_ttree_array_entry.subtree_leaves_with_path(src_ref);

        // Get leaves of the dst aggregate node
        let dst_ttree_id = self.agg_identifier_to_id.get(dst_identity).expect("to exist");
        let dst_ttree = self.ttrees.get(dst_ttree_id.to_usize()).expect("to exist");
        let dst_leaves = dst_ttree.subtree_leaves_with_path(dst_ref);

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        for (src_path_key, src_ttree_leaf_id) in src_leaves.iter() {
            let src_ttree_leaf = src_ttree_array_entry.graph.node_weight(*src_ttree_leaf_id).unwrap();

            // If there is a matching path in the dst aggregate node, add an edge
            if dst_leaves.contains_key(src_path_key) {
                let dst_ttree_leaf_id = dst_leaves.get(src_path_key).unwrap();
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
                panic!("Not connected src_ref {:?}\nsrc_identity {:?}\nsrc_leaves {:?}\ndst_ref {:?}\ndst_identity {:?}\ndst_leaves {:?}",
                    src_ref, src_identity, src_leaves, dst_ref, dst_identity, dst_leaves);
            }
        }

        for edge in edges {
            self.add_edge(edge.0, edge.1, edge.2);
        }
    }

    pub fn check_node_type(&self) {
        for ttree in self.ttrees.iter() {
            let leaf_ids = ttree.all_leaves();
            let mut prev_nt_opt: Option<RippleNodeType> = None;
            for leaf_id in leaf_ids {
                let leaf = ttree.graph.node_weight(leaf_id).unwrap();
                let ir_id = leaf.id.unwrap();
                let ir_node = self.graph.node_weight(ir_id).unwrap();
                if let Some(prev_nt) = prev_nt_opt.clone() {
                    if ir_node.tpe != prev_nt {
                        ttree.print_tree();
                        panic!("Nodes under the same ttree has different types {:?} {:?}", ir_node, prev_nt);
                    }
                } else {
                    prev_nt_opt = Some(ir_node.tpe.clone());
                }
            }
        }
    }

    /// Similar to node_indices in petgraph.
    /// Here, `AggNodeIndex` represents an aggregate node index
    pub fn node_indices_agg(&self) -> Vec<AggNodeIndex> {
        (0..self.ttrees.len()).map(|x| AggNodeIndex::from(x)).collect()
    }

    pub fn node_weight_agg(&self, id: AggNodeIndex) -> (Option<&TypeTree>, Option<&AggNodeIdentifier>) {
        (self.ttrees.get(id.to_usize()), self.agg_id_to_identifier.get(&id))
    }

    pub fn neighbors_agg(&self, id: AggNodeIndex) -> Vec<AggNodeIndex> {
        let ttree = self.ttrees.get(id.to_usize()).unwrap();
        let leaf_ids = ttree.all_leaves();

        let mut neighbor_ttree_ids: IndexSet<AggNodeIndex> = IndexSet::new();
        for leaf_id in leaf_ids {
            let leaf = ttree.graph.node_weight(leaf_id).unwrap();
            let ir_id = leaf.id.unwrap();

            let neighbor_ids = self.graph.neighbors_directed(ir_id, Outgoing);
            for nid in neighbor_ids {
                let neighbor_agg_leaf = self.id_to_agg_leaf.get(&nid).unwrap();
                neighbor_ttree_ids.insert(neighbor_agg_leaf.agg_id);
            }

            let neighbor_ids = self.graph.neighbors_directed(ir_id, Incoming);
            for nid in neighbor_ids {
                let neighbor_agg_leaf = self.id_to_agg_leaf.get(&nid).unwrap();
                neighbor_ttree_ids.insert(neighbor_agg_leaf.agg_id);
            }
        }
        return neighbor_ttree_ids.iter().map(|x| *x).collect();
    }

    pub fn neighbors_directed_agg(&self, id: AggNodeIndex, dir: petgraph::Direction) -> Vec<AggNodeIndex> {
        let ttree = self.ttrees.get(id.to_usize()).unwrap();
        let leaf_ids = ttree.all_leaves();

        let mut neighbor_ttree_ids: IndexSet<AggNodeIndex> = IndexSet::new();
        for leaf_id in leaf_ids {
            let leaf = ttree.graph.node_weight(leaf_id).unwrap();
            let ir_id = leaf.id.unwrap();

            let neighbor_ids = self.graph.neighbors_directed(ir_id, dir);
            for nid in neighbor_ids {
                let neighbor_agg_leaf = self.id_to_agg_leaf.get(&nid).unwrap();
                neighbor_ttree_ids.insert(neighbor_agg_leaf.agg_id);
            }
        }
        return neighbor_ttree_ids.iter().map(|x| *x).collect();
    }

    pub fn edges_directed_agg(&self, id: AggNodeIndex, dir: petgraph::Direction) -> IndexMap<AggEdgeIdentifier, Vec<EdgeIndex>> {
        let ttree = self.ttrees.get(id.to_usize()).unwrap();
        let leaf_ids = ttree.all_leaves();

        let mut ttree_edge_map: IndexMap<AggEdgeIdentifier, Vec<EdgeIndex>> = IndexMap::new();
        for leaf_id in leaf_ids {
            let leaf = ttree.graph.node_weight(leaf_id).unwrap();
            let ir_id = leaf.id.unwrap();

            let edge_ids = self.graph.edges_directed(ir_id, dir);
            for eid in edge_ids {
                let (_src, dst) = self.graph.edge_endpoints(eid.id()).unwrap();
                let neighbor_agg_id = self.id_to_agg_leaf.get(&dst).unwrap();
                let rir_edge = self.graph.edge_weight(eid.id()).unwrap();

                let edge_map_key = AggEdgeIdentifier::new(neighbor_agg_id.agg_id, rir_edge.et.clone());
                if !ttree_edge_map.contains_key(&edge_map_key) {
                    ttree_edge_map.insert(edge_map_key.clone(), vec![]);
                }
                ttree_edge_map
                    .get_mut(&edge_map_key)
                    .unwrap()
                    .push(eid.id());
            }
        }
        return ttree_edge_map;
    }

    fn add_flat_edge_to_agg_edge(
        &self,
        id: NodeIndex,
        eid: EdgeIndex,
        ttree_edge_map: &mut IndexMap<AggEdgeIdentifier, Vec<EdgeIndex>>
    ) {
        let neighbor_agg_id = self.id_to_agg_leaf.get(&id).unwrap();
        let rir_edge = self.graph.edge_weight(eid).unwrap();

        let edge_map_key = AggEdgeIdentifier::new(neighbor_agg_id.agg_id, rir_edge.et.clone());
        if !ttree_edge_map.contains_key(&edge_map_key) {
            ttree_edge_map.insert(edge_map_key.clone(), vec![]);
        }
        ttree_edge_map
            .get_mut(&edge_map_key)
            .unwrap()
            .push(eid);
    }

    /// Returns undirected aggregate edges
    pub fn edges_agg(&self, id: AggNodeIndex) -> IndexMap<AggEdgeIdentifier, Vec<EdgeIndex>> {
        let ttree = self.ttrees.get(id.to_usize()).unwrap();
        let leaf_ids = ttree.all_leaves();

        let mut ttree_edge_map: IndexMap<AggEdgeIdentifier, Vec<EdgeIndex>> = IndexMap::new();
        for leaf_id in leaf_ids {
            let leaf = ttree.graph.node_weight(leaf_id).unwrap();
            let ir_id = leaf.id.unwrap();

            let edge_ids = self.graph.edges_directed(ir_id, Outgoing);
            for eid in edge_ids {
                let (_src, dst) = self.graph.edge_endpoints(eid.id()).unwrap();
                self.add_flat_edge_to_agg_edge(dst, eid.id(), &mut ttree_edge_map);
            }
            let edge_ids = self.graph.edges_directed(ir_id, Incoming);
            for eid in edge_ids {
                let (src, _dst) = self.graph.edge_endpoints(eid.id()).unwrap();
                self.add_flat_edge_to_agg_edge(src, eid.id(), &mut ttree_edge_map);
            }
        }
        return ttree_edge_map;
    }

    pub fn vismap_agg(&self) -> AggVisMap {
        AggVisMap::new(self.ttrees.len() as u32)
    }
}

#[derive(Debug, Clone)]
pub struct RippleIR {
    pub name: Identifier,
    pub graphs: IndexMap<Identifier, RippleGraph>,
    pub hierarchy: Hierarchy,
}

impl RippleIR {
    pub fn new(name: Identifier) -> Self {
        Self {
            name,
            graphs: IndexMap::new(),
            hierarchy: Hierarchy::default(),
        }
    }
}

impl GraphViz for RippleGraph {
    fn graphviz_string(
        self: &Self,
        node_attr: Option<&NodeAttributeMap>,
        edge_attr: Option<&EdgeAttributeMap>
    ) -> Result<String, std::io::Error> {
        use graphviz_rust::{
            attributes::{rankdir, EdgeAttributes, GraphAttributes, NodeAttributes},
            dot_generator::{edge, id, node_id},
            dot_structures::Id,
            dot_structures::Stmt as DotStmt,
            dot_structures::Subgraph as DotSubgraph,
            dot_structures::Node as DotNode,
            dot_structures::NodeId as DotNodeId,
            dot_structures::*,
            printer::{DotPrinter, PrinterContext}
        };

        let mut g = graphviz_rust::dot_structures::Graph::DiGraph {
            id: graphviz_rust::dot_generator::id!(""),
            strict: false,
            stmts: vec![
                DotStmt::from(GraphAttributes::rankdir(rankdir::TB)),
                DotStmt::from(GraphAttributes::splines(true)),
                DotStmt::from(GraphAttributes::mindist(2.0)),
                DotStmt::from(GraphAttributes::ranksep(5.0)),
            ]
        };

        // Add nodes
        for (ttree_idx, ttree) in self.ttrees.iter().enumerate() {
            let leaves = ttree.all_leaves();
            let root_info = self.agg_id_to_identifier.get(&AggNodeIndex::from(ttree_idx)).unwrap();

            // Create graphviz subgraph to group nodes together
            let subgraph_name = format!("\"cluster_{}_{}\"",
                root_info.name,
                ttree_idx).replace('"', "");
            let mut subgraph = DotSubgraph {
                id: Id::Plain(subgraph_name),
                stmts: vec![]
            };

            // Collect all flattened nodes under the current aggregate node
            for ttree_id in leaves.iter() {
                let leaf = ttree.graph.node_weight(*ttree_id).unwrap();
                let rir_id = leaf.id.unwrap();
                let rir_node = self.graph.node_weight(rir_id).unwrap();

                let node_label_inner = format!("{}", rir_node).to_string().replace('"', "");
                let node_label = format!("\"{}\"", node_label_inner);


                // Create graphviz node
                let mut gv_node = DotNode {
                    id: DotNodeId(Id::Plain(rir_id.index().to_string()), None),
                    attributes: vec![
                        NodeAttributes::label(node_label)
                    ],
                };

                // Add node attribute if it exists
                if let Some(na) = node_attr {
                    if na.contains_key(&rir_id) {
                        gv_node.attributes.push(na.get(&rir_id).unwrap().clone());
                    }
                }
                subgraph.stmts.push(DotStmt::from(gv_node));
            }
            g.add_stmt(DotStmt::from(DotSubgraph::from(subgraph)));
        }

        // Add edges
        for eid in self.graph.edge_indices() {
            let ep = self.graph.edge_endpoints(eid).unwrap();
            let w = self.graph.edge_weight(eid).unwrap();

            // Create graphviz edge
            let mut e = edge!(
                node_id!(ep.0.index().to_string()) =>
                node_id!(ep.1.index().to_string()));

            let edge_label_inner = format!("{}", w).to_string().replace('"', "");
            let edge_label = format!("\"{}\"", edge_label_inner);
            e.attributes.push(EdgeAttributes::label(edge_label));

            // Add edge attribute if it exists
            if let Some(ea) = edge_attr {
                if ea.contains_key(&eid) {
                    e.attributes.push(ea.get(&eid).unwrap().clone());
                }
            }

            g.add_stmt(Stmt::Edge(e));
        }

        // Export to pdf
        let dot = g.print(&mut PrinterContext::new(true, 4, "\n".to_string(), 90));
        Ok(dot)
    }
}
