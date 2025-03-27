use fixedbitset::FixedBitSet;
use chirrtl_parser::ast::*;
use crate::ir::rir::rgraph::*;
use crate::ir::typetree::typetree::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggEdge {
    pub dst_id: AggNodeIndex,
    pub et: RippleEdgeType,
}

impl AggEdge {
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

    /// `NodeIndex` of the `TypeTree` leaf node that this graph node corresponds to
    pub leaf_id: TTreeNodeIndex,
}

impl AggNodeLeafIndex {
    pub fn new(agg_id: AggNodeIndex, leaf_id: TTreeNodeIndex) -> Self {
        Self { agg_id, leaf_id }
    }
}

/// Can be used as a key to identify a `TypeTree` in `RippleGraph`
/// Represents a unique aggregate node in the IR
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggNode {
    /// Identifier of the reference root
    pub name: Identifier,

    /// Need to identify the type of the node as multiple nodes can
    /// use the same Identifier but have different `RippleNodeType`.
    /// E.g., Phi and Reg nodes
    pub nt: RippleNodeType,
}

impl AggNode {
    pub fn new(name: Identifier, nt: RippleNodeType) -> Self {
        Self { name, nt }
    }
}
