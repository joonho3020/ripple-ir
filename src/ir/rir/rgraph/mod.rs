pub mod rgraph_graphviz;
pub mod rgraph_impl;

use crate::ir::rir::rnode::*;
use crate::ir::rir::redge::*;
use crate::ir::IndexGen;
use crate::ir::rir::agg::*;
use petgraph::graph::Graph;
use petgraph::graph::NodeIndex;
use bimap::BiMap;
use indexmap::IndexMap;

pub type FlatRippleGraph = Graph<RippleNode, RippleEdge>;

#[derive(Debug, Clone)]
pub struct RippleGraph {
    /// Used to generate unique `AggNodeIndex`
    agg_node_idx_gen: IndexGen,

    /// Used to generate unique `RippleNodeIndex`
    node_idx_gen: IndexGen,

    /// Used to generate unique `AggEdgeIndex`
    agg_edge_idx_gen: IndexGen,

    /// Used to generate unique `RippleEdgeIndex`
    edge_idx_gen: IndexGen,

    /// Graph of this IR
    graph: FlatRippleGraph,

    /// Bi-directional map that ties a low level graph node to 
    /// each aggregate node and vice-versa
    agg_node_map: BiMap<RippleNodeIndex, AggNodeLeafIndex>,

    /// Map that contains metadata for each unique aggregate node
    agg_nodes: IndexMap<AggNodeIndex, AggNodeData>,

    /// Map that ties a low level graph edge to each aggregate edge
    agg_edge_map: IndexMap<AggEdgeIndex, Vec<RippleEdgeIndex>>,

    /// Map that contains metadata for each unique aggregate edges
    /// - This is bidirectional
    agg_neighbors: IndexMap<AggNodeIndex, Vec<AggEdge>>,

    /// Cache that maps graph nodes to its aggregate node.
    /// - Must be updated correctly when removing nodes, or just invalidated and
    /// reconstructed from scratch
    /// - This is because of how petgraph changes the `NodeIndex` when removing nodes.
    node_map_cache: BiMap<NodeIndex, AggNodeLeafIndex>,
}
