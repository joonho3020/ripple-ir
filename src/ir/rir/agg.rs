use fixedbitset::FixedBitSet;
use rusty_firrtl::*;
use crate::define_index_type;
use crate::ir::rir::rnode::*;
use crate::ir::rir::redge::*;
use crate::ir::typetree::tnode::TypeTreeNodeIndex;
use crate::ir::typetree::typetree::*;

/// Represents a unique aggregate node in the IR
/// - Contains metadata to tie a group of flat nodes in the graph as a single
/// aggregate node
#[derive(Debug, Clone)]
pub struct AggNodeData {
    /// Identifier of the reference root
    pub name: Identifier,

    /// Need to identify the type of the node as multiple nodes can
    /// use the same Identifier but have different `RippleNodeType`.
    /// E.g., Phi and Reg nodes
    pub nt: RippleNodeType,

    /// Used to tie nodew within a aggregate node together
    pub ttree: Option<TypeTree>,
}

impl AggNodeData {
    pub fn new(name: Identifier, nt: RippleNodeType, ttree: Option<TypeTree>) -> Self {
        Self { name, nt, ttree }
    }
}

define_index_type!(AggNodeIndex);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AggNodeLeafIndex {
    /// Aggregate node id
    pub agg_id: AggNodeIndex,

    /// Unique index of the `TypeTree` leaf node that this graph node corresponds to
    pub leaf_id: TypeTreeNodeIndex
}

impl AggNodeLeafIndex {
    pub fn new(agg_id: AggNodeIndex, leaf_id: TypeTreeNodeIndex) -> Self {
        Self { agg_id, leaf_id }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggEdgeData {
    pub et: RippleEdgeType,
}

impl AggEdgeData {
    pub fn new(et: RippleEdgeType) -> Self {
        Self { et }
    }
}

/// Represents a unique aggregate edge in the IR
/// - Contains metadata to tie a group of flat edges in the graph as a single
/// aggregate edge
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggEdge {
    pub id: AggEdgeIndex,
    pub data: AggEdgeData,
    pub src: AggNodeIndex,
    pub dst: AggNodeIndex,
    pub src_subtree_root: TypeTreeNodeIndex,
    pub dst_subtree_root: TypeTreeNodeIndex,
}

impl AggEdge {
    pub fn new(
        id: AggEdgeIndex,
        data: AggEdgeData,
        src: AggNodeIndex,
        dst: AggNodeIndex,
        src_subtree_root: TypeTreeNodeIndex,
        dst_subtree_root: TypeTreeNodeIndex,
    ) -> Self {
        Self { id, data, src, dst, src_subtree_root, dst_subtree_root }
    }
}

define_index_type!(AggEdgeIndex);

/// A datastructure used for tracking which aggregate nodes have been visited
/// during a traversal
#[derive(Debug, Clone)]
pub struct AggVisMap {
    visited: FixedBitSet
}

impl AggVisMap {
    pub fn new(num_bits: u32) -> Self {
        Self { visited: FixedBitSet::with_capacity(num_bits as usize) }
    }

    /// Already visited agg-node with `id`
    pub fn is_visited(&self, id: AggNodeIndex) -> bool {
        self.visited.contains(id.into())
    }

    /// Visit aggregate node with `id`
    pub fn visit(&mut self, id: AggNodeIndex) {
        self.visited.set(id.into(), true);
    }

    /// Graph contains some unvisited agg-nodes
    pub fn has_unvisited(&self) -> bool {
        self.visited.count_zeroes(..) > 0
    }

    /// List of indices of unvisited agg-nodes
    pub fn unvisited_ids(&self) -> Vec<AggNodeIndex> {
        self.visited.zeroes().map(|x| x.into()).collect()
    }
}
