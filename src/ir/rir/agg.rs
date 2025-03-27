use fixedbitset::FixedBitSet;
use chirrtl_parser::ast::*;
use crate::define_index_type;
use crate::ir::rir::rnode::*;
use crate::ir::rir::redge::*;
use crate::ir::typetree::typetree::*;

/// Can be used as a key to identify a `TypeTree` in `RippleGraph`
/// Represents a unique aggregate node in the IR
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggNodeLeafIndex {
    /// Aggregate node id
    pub agg_id: AggNodeIndex,

    /// `NodeIndex` of the `TypeTree` leaf node that this graph node corresponds to
    pub leaf_id: TTreeNodeIndex,
}

impl AggNodeLeafIndex {
    pub fn new(agg_id: AggNodeIndex, leaf_id: TTreeNodeIndex) -> Self {
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggEdge {
    pub id: AggEdgeIndex,
    pub data: AggEdgeData,
    pub src: AggNodeIndex,
    pub dst: AggNodeIndex,
    pub src_subtree_root: TTreeNodeIndex,
    pub dst_subtree_root: Option<TTreeNodeIndex>,
}

impl AggEdge {
    pub fn new(
        id: AggEdgeIndex,
        data: AggEdgeData,
        src: AggNodeIndex,
        dst: AggNodeIndex,
        src_subtree_root: TTreeNodeIndex,
        dst_subtree_root: Option<TTreeNodeIndex>,
    ) -> Self {
        Self { id, data, src, dst, src_subtree_root, dst_subtree_root }
    }
}

define_index_type!(AggEdgeIndex);

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
