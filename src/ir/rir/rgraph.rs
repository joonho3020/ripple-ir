use chirrtl_parser::ast::*;
use derivative::Derivative;
use petgraph::visit::EdgeRef;
use petgraph::Direction::Incoming;
use petgraph::Direction::Outgoing;
use std::hash::Hash;
use indexmap::{IndexMap, IndexSet};
use petgraph::graph::{Graph, NodeIndex, EdgeIndex};
use bimap::BiMap;
use crate::common::graphviz::*;
use crate::ir::whentree::Condition;
use crate::ir::typetree::typetree::*;
use crate::ir::firir::*;
use crate::ir::rir::agg::*;
use crate::ir::*;
use crate::impl_clean_display;


#[derive(Derivative, Clone, PartialEq, Eq, Hash)]
#[derivative(Debug)]
pub struct RippleNodeInfo {
    pub name: Option<Identifier>,
    pub tpe: RippleNodeType,
    pub tg: GroundType
}

impl RippleNodeInfo {
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
impl_clean_display!(RippleNodeInfo);

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

#[derive(Derivative, Clone, PartialEq, Eq, Hash)]
pub struct RippleNode {
    pub info: RippleNodeInfo,
    pub id: Ripple
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

type IRGraph = Graph<RippleNode, RippleEdge>;

#[derive(Debug, Clone)]
pub struct RippleGraph {
    agg_node_idx_gen: IndexGen,
    agg_edge_idx_gen: IndexGen,
    node_idx_gen: IndexGen,
    edge_idx_gen: IndexGen,

    /// Graph of this IR
    pub graph: IRGraph,

    /// Bi-directional map that ties a low level `graph` node to 
    /// each aggregate node and vice-versa
    pub flatid_aggleaf_bimap: BiMap<NodeIndex, AggNodeLeafIndex>,

    /// map that contains metadata for each unique aggregate node
    pub agg_nodes: Vec<AggNode>,

    pub agg_edges: Vec<AggEdge>,
}

impl RippleGraph {
    pub fn new() -> Self {
        Self {
            agg_node_idx_gen: IndexGen::new(),
            agg_edge_idx_gen: IndexGen::new(),
            node_idx_gen: IndexGen::new(),
            edge_idx_gen: IndexGen::new(),
            graph: IRGraph::new(),
            flatid_aggleaf_bimap: BiMap::new(),
            agg_nodes: vec![],
            agg_edges: vec![],
        }
    }

    pub fn check_metadata_consistency(&self) {
        todo!("Check metadata consistency needs to be implemented");
    }

    /// Removes a single low-level node from the IR
    pub fn remove_node(&mut self, id: NodeIndex) {
        let last_id = NodeIndex::new(self.graph.node_count() - 1);

        // Remove node from the graph
        self.graph.remove_node(id);

        // Remove id from flatid_aggleaf_bimap
        if self.flatid_aggleaf_bimap.contains_left(&id) {
            self.flatid_aggleaf_bimap.remove_by_left(&id).unwrap();
        }

        // Petgraph moves the last node to the removed node position
        if last_id != id {
            // Remove the last id from flatid_aggleaf_bimap
            let last_agg_leaf = self.flatid_aggleaf_bimap.remove_by_left(&last_id).unwrap().1;

            // Map id to last_agg_leaf
            self.flatid_aggleaf_bimap.insert(id, last_agg_leaf);
        }
    }

    pub fn add_node_agg(&mut self, node: AggNode) -> AggNodeIndex {
        let my_ttree = node.ttree.view().unwrap();
        let agg_id = self.agg_nodes.len() as u32;
        let leaves = my_ttree.leaves();

        // Add all the leaf nodes
        for leaf_id in leaves.iter() {
            let leaf = my_ttree.get_node(*leaf_id).unwrap();
            let tg = match &leaf.tpe {
                TypeTreeNodeType::Ground(x) => x,
                _ => {
                    // Empty aggregate type, add a invalid ground type
                    &GroundType::Invalid
                }
            };

            let leaf_name = my_ttree.node_name(&node.name, *leaf_id);

            // Insert new graph node
            let rgnode = RippleNode::new(Some(leaf_name), node.nt.clone(), tg.clone());
            let rg_id = self.graph.add_node(rgnode);

            let anli = AggNodeLeafIndex::new(AggNodeIndex::from(agg_id), *leaf_id);
            let prev_len = self.flatid_aggleaf_bimap.len();

            // Add this node to the flatid_aggleaf_bimap
            self.flatid_aggleaf_bimap.insert(
                rg_id,
                anli);

            let after_len = self.flatid_aggleaf_bimap.len();
            assert!(after_len == prev_len + 1);
        }

        // Add the type tree
        self.agg_nodes.push(node);
        return AggNodeIndex::from(agg_id);
    }

    pub fn flatid(&self, aggid: AggNodeIndex, leafid: TTreeNodeIndex) -> Option<&NodeIndex> {
        let aggleafidx = AggNodeLeafIndex::new(aggid, leafid);
        self.flatid_aggleaf_bimap.get_by_right(&aggleafidx)
    }

    pub fn add_fanout_edge_agg(&mut self, src: AggNodeIndex, dst: AggNodeIndex, edge: AggEdge) {
    }

    pub fn add_edge_agg(
        &mut self,
        src_agg_id: AggNodeIndex,
        dst_agg_id: AggNodeIndex,
        src_ref: &Reference,
        dst_ref: &Reference,
        edge: AggEdge
    ) -> AggEdgeIndex {
        let src_agg_node = self.agg_nodes.get(src_agg_id.to_usize()).unwrap();
        let src_ttree = src_agg_node.ttree.view().unwrap();
        let src_leaves = src_ttree.subtree_leaves_with_path(src_ref);

        let dst_agg_node = self.agg_nodes.get(dst_agg_id.to_usize()).unwrap();
        let dst_ttree = dst_agg_node.ttree.view().unwrap();
        let dst_leaves = dst_ttree.subtree_leaves_with_path(dst_ref);

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        for (src_path_identity, src_ttree_leaf_id) in src_leaves.iter() {
            let src_ttree_leaf = src_ttree.get_node(*src_ttree_leaf_id).unwrap();
            let src_flatid = self.flatid(src_agg_id, *src_ttree_leaf_id).unwrap();

            // If there is a matching path in the dst aggregate node, add an edge
            if dst_leaves.contains_key(src_path_identity) {
                let dst_ttree_leaf_id = dst_leaves.get(src_path_identity).unwrap();
                let dst_flatid = self.flatid(dst_agg_id, *dst_ttree_leaf_id).unwrap();
                if src_ttree_leaf.dir == TypeDirection::Outgoing {
                    edges.push((*src_flatid, *dst_flatid, RippleEdge::new(None, edge.et.clone())));
                } else {
                    edges.push((*dst_flatid, *src_flatid, RippleEdge::new(None, edge.et.clone())));
                }
            } else {
                panic!("Not connected src_ref {:?}\nsrc_id {:?}\nsrc_leaves {:?}\ndst_ref {:?}\ndst_identity {:?}\ndst_leaves {:?}",
                    src_ref, src_agg_id, src_leaves, dst_ref, dst_agg_id, dst_leaves);
            }
        }

        for edge in edges {
            self.graph.add_edge(edge.0, edge.1, edge.2);
        }
    }

    pub fn add_single_edge(
        &mut self,
        src_identity: &AggNode,
        src_ref: &Reference,
        dst_identity: &AggNode,
        dst_ref: &Reference,
        et: RippleEdgeType
    ) {
        let src_aggid = self.aggidty_aggid_bimap.get_by_left(src_identity).expect("to exist");
        let src_ttree = self.ttrees.get(src_aggid.to_usize()).expect("to exist").view().unwrap();
        let src_leaves = src_ttree.subtree_leaves_with_path(src_ref);

        let dst_aggid = self.aggidty_aggid_bimap.get_by_left(dst_identity).expect("to exist");
        let dst_ttree = self.ttrees.get(dst_aggid.to_usize()).expect("to exist").view().unwrap();
        let dst_leaves = dst_ttree.subtree_leaves_with_path(dst_ref);

        assert!(src_leaves.len() == 1, "add_single_edge got multiple src_leaves {:?}", src_leaves);
        assert!(dst_leaves.len() > 0, "add_single_edge got zero destinations");

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        let (_src_path_identity, src_ttree_leaf_id) = src_leaves.first().unwrap();
        let src_ttree_leaf = src_ttree.get_node(*src_ttree_leaf_id).unwrap();
        let src_flatid = self.flatid(*src_aggid, *src_ttree_leaf_id).unwrap();

        for (_dst_path_identity, dst_ttree_leaf_id) in dst_leaves {
            let dst_flatid = self.flatid(*dst_aggid, dst_ttree_leaf_id).unwrap();
            if src_ttree_leaf.dir == TypeDirection::Outgoing {
                edges.push((*src_flatid, *dst_flatid, RippleEdge::new(None, et.clone())));
            } else {
                edges.push((*dst_flatid, *src_flatid, RippleEdge::new(None, et.clone())));
            }
        }

        for edge in edges {
            self.graph.add_edge(edge.0, edge.1, edge.2);
        }
    }

    pub fn add_aggregate_edge(
        &mut self,
        src_identity: &AggNode,
        src_ref: &Reference,
        dst_identity: &AggNode,
        dst_ref: &Reference,
        et: RippleEdgeType
    ) {
        // Get leaves of the src aggregate node
        let src_aggid = self.aggidty_aggid_bimap.get_by_left(src_identity).expect("to exist");
        let src_ttree = self.ttrees.get(src_aggid.to_usize()).expect("to exist").view().unwrap();
        let src_leaves = src_ttree.subtree_leaves_with_path(src_ref);

        // Get leaves of the dst aggregate node
        let dst_aggid = self.aggidty_aggid_bimap.get_by_left(dst_identity).expect("to exist");
        let dst_ttree = self.ttrees.get(dst_aggid.to_usize()).expect("to exist").view().unwrap();
        let dst_leaves = dst_ttree.subtree_leaves_with_path(dst_ref);

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        for (src_path_identity, src_ttree_leaf_id) in src_leaves.iter() {
            let src_ttree_leaf = src_ttree.get_node(*src_ttree_leaf_id).unwrap();
            let src_flatid = self.flatid(*src_aggid, *src_ttree_leaf_id).unwrap();

            // If there is a matching path in the dst aggregate node, add an edge
            if dst_leaves.contains_key(src_path_identity) {
                let dst_ttree_leaf_id = dst_leaves.get(src_path_identity).unwrap();
                let dst_flatid = self.flatid(*dst_aggid, *dst_ttree_leaf_id).unwrap();
                if src_ttree_leaf.dir == TypeDirection::Outgoing {
                    edges.push((*src_flatid, *dst_flatid, RippleEdge::new(None, et.clone())));
                } else {
                    edges.push((*dst_flatid, *src_flatid, RippleEdge::new(None, et.clone())));
                }
            } else {
                println!("EdgeType {:?}", et);
                panic!("Not connected src_ref {:?}\nsrc_identity {:?}\nsrc_leaves {:?}\ndst_ref {:?}\ndst_identity {:?}\ndst_leaves {:?}",
                    src_ref, src_identity, src_leaves, dst_ref, dst_identity, dst_leaves);
            }
        }

        for edge in edges {
            self.graph.add_edge(edge.0, edge.1, edge.2);
        }
    }

    pub fn add_aggregate_mem_edge(
        &mut self,
        src_identity: &AggNode,
        src_ref: &Reference,
        dst_identity: &AggNode,
        dst_ref: &Reference,
        et: RippleEdgeType
    ) {
        // Get leaves of the src aggregate node
        let src_aggid = self.aggidty_aggid_bimap.get_by_left(src_identity).expect("to exist");
        let src_ttree = self.ttrees.get(src_aggid.to_usize()).expect("to exist").view().unwrap();
        let src_ttree_array_entry = src_ttree.subtree_array_element();
        let src_leaves = src_ttree_array_entry.subtree_leaves_with_path(src_ref);

        // Get leaves of the dst aggregate node
        let dst_aggid = self.aggidty_aggid_bimap.get_by_left(dst_identity).expect("to exist");
        let dst_ttree = self.ttrees.get(dst_aggid.to_usize()).expect("to exist").view().unwrap();
        let dst_leaves = dst_ttree.subtree_leaves_with_path(dst_ref);

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        for (src_path_key, src_ttree_leaf_id) in src_leaves.iter() {
            let src_ttree_leaf = src_ttree_array_entry.get_node(*src_ttree_leaf_id).unwrap();
            let src_flatid = self.flatid(*src_aggid, *src_ttree_leaf_id).expect(&format!("WTF src_ref: {:?} src_identity: {:?} src_agg_id: {:?} src_leaf_id: {:?} {:#?}", src_ref, src_identity, src_aggid, src_ttree_leaf_id, self.flatid_aggleaf_bimap));

            // If there is a matching path in the dst aggregate node, add an edge
            if dst_leaves.contains_key(src_path_key) {
                let dst_ttree_leaf_id = dst_leaves.get(src_path_key).unwrap();
                let dst_flatid = self.flatid(*dst_aggid, *dst_ttree_leaf_id).unwrap();
                if src_ttree_leaf.dir == TypeDirection::Outgoing {
                    edges.push((*src_flatid, *dst_flatid, RippleEdge::new(None, et.clone())));
                } else {
                    edges.push((*dst_flatid, *src_flatid, RippleEdge::new(None, et.clone())));
                }
            } else {
                panic!("Not connected src_ref {:?}\nsrc_identity {:?}\nsrc_leaves {:?}\ndst_ref {:?}\ndst_identity {:?}\ndst_leaves {:?}",
                    src_ref, src_identity, src_leaves, dst_ref, dst_identity, dst_leaves);
            }
        }

        for edge in edges {
            self.graph.add_edge(edge.0, edge.1, edge.2);
        }
    }

    pub fn check_node_type(&self) {
        for (aggid, ttree) in self.ttrees.iter().enumerate() {
            let leaf_ids = ttree.view().unwrap().leaves();
            let mut prev_nt_opt: Option<RippleNodeType> = None;
            for leaf_id in leaf_ids {
                let aggleafid = AggNodeLeafIndex::new(AggNodeIndex::from(aggid), leaf_id);
                let flatid = self.flatid_aggleaf_bimap.get_by_right(&aggleafid).unwrap();

                let ir_node = self.graph.node_weight(*flatid).unwrap();
                if let Some(prev_nt) = prev_nt_opt.clone() {
                    if ir_node.tpe != prev_nt {
                        ttree.view().unwrap().print_tree();
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
        (0..self.agg_nodes.len()).map(|x| AggNodeIndex::from(x)).collect()
    }

    pub fn node_weight_agg(&self, id: AggNodeIndex) -> Option<&AggNode> {
        self.agg_nodes.get(id.to_usize())
    }

    pub fn neighbors_agg(&self, agg_id: AggNodeIndex) -> Vec<AggNodeIndex> {
        let agg_node = self.agg_nodes.get(agg_id.to_usize()).unwrap();
        let ttree = &agg_node.ttree;
        let leaf_ids = ttree.view().unwrap().leaves();

        let mut neighbor_ttree_ids: IndexSet<AggNodeIndex> = IndexSet::new();
        for leaf_id in leaf_ids {
            let ir_id = *self.flatid(agg_id, leaf_id).unwrap();
            let neighbor_ids = self.graph.neighbors_directed(ir_id, Outgoing);

            for nid in neighbor_ids {
                let neighbor_agg_leaf = self.flatid_aggleaf_bimap.get_by_left(&nid).unwrap();
                neighbor_ttree_ids.insert(neighbor_agg_leaf.agg_id);
            }

            let neighbor_ids = self.graph.neighbors_directed(ir_id, Incoming);
            for nid in neighbor_ids {
                let neighbor_agg_leaf = self.flatid_aggleaf_bimap.get_by_left(&nid).unwrap();
                neighbor_ttree_ids.insert(neighbor_agg_leaf.agg_id);
            }
        }
        return neighbor_ttree_ids.iter().map(|x| *x).collect();
    }

    pub fn neighbors_directed_agg(&self, agg_id: AggNodeIndex, dir: petgraph::Direction) -> Vec<AggNodeIndex> {
        let agg_node = self.agg_nodes.get(agg_id.to_usize()).unwrap();
        let ttree = &agg_node.ttree;
        let leaf_ids = ttree.view().unwrap().leaves();

        let mut neighbor_ttree_ids: IndexSet<AggNodeIndex> = IndexSet::new();
        for leaf_id in leaf_ids {
            let ir_id = *self.flatid(agg_id, leaf_id).unwrap();
            let neighbor_ids = self.graph.neighbors_directed(ir_id, dir);

            for nid in neighbor_ids {
                let neighbor_agg_leaf = self.flatid_aggleaf_bimap.get_by_left(&nid).unwrap();
                neighbor_ttree_ids.insert(neighbor_agg_leaf.agg_id);
            }
        }
        return neighbor_ttree_ids.iter().map(|x| *x).collect();
    }

    pub fn edges_directed_agg(&self, agg_id: AggNodeIndex, dir: petgraph::Direction) -> IndexMap<AggEdge, Vec<EdgeIndex>> {
        let ttree = self.ttrees.get(agg_id.to_usize()).unwrap();
        let leaf_ids = ttree.view().unwrap().leaves();

        let mut ttree_edge_map: IndexMap<AggEdge, Vec<EdgeIndex>> = IndexMap::new();
        for leaf_id in leaf_ids {
            let ir_id = *self.flatid(agg_id, leaf_id).unwrap();
            let edge_ids = self.graph.edges_directed(ir_id, dir);

            for eid in edge_ids {
                let (_src, dst) = self.graph.edge_endpoints(eid.id()).unwrap();
                let neighbor_agg_id = self.flatid_aggleaf_bimap.get_by_left(&dst).unwrap();
                let rir_edge = self.graph.edge_weight(eid.id()).unwrap();

                let edge_map_key = AggEdge::new(neighbor_agg_id.agg_id, rir_edge.et.clone());
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
        ttree_edge_map: &mut IndexMap<AggEdge, Vec<EdgeIndex>>
    ) {
        let neighbor_agg_id = self.flatid_aggleaf_bimap.get_by_left(&id).unwrap();
        let rir_edge = self.graph.edge_weight(eid).unwrap();

        let edge_map_key = AggEdge::new(neighbor_agg_id.agg_id, rir_edge.et.clone());
        if !ttree_edge_map.contains_key(&edge_map_key) {
            ttree_edge_map.insert(edge_map_key.clone(), vec![]);
        }
        ttree_edge_map
            .get_mut(&edge_map_key)
            .unwrap()
            .push(eid);
    }

    /// Returns undirected aggregate edges
    pub fn edges_agg(&self, agg_id: AggNodeIndex) -> IndexMap<AggEdge, Vec<EdgeIndex>> {
        let ttree = self.ttrees.get(agg_id.to_usize()).unwrap();
        let leaf_ids = ttree.view().unwrap().leaves();

        let mut ttree_edge_map: IndexMap<AggEdge, Vec<EdgeIndex>> = IndexMap::new();
        for leaf_id in leaf_ids {
            let ir_id = *self.flatid(agg_id, leaf_id).unwrap();

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
        for (agg_id, ttree) in self.ttrees.iter().enumerate() {
            let leaves = ttree.view().unwrap().leaves();
            let root_info = self.aggidty_aggid_bimap.get_by_right(&AggNodeIndex::from(agg_id)).unwrap();

            // Create graphviz subgraph to group nodes together
            let subgraph_name = format!("\"cluster_{}_{}\"",
                root_info.name,
                agg_id).replace('"', "");
            let mut subgraph = DotSubgraph {
                id: Id::Plain(subgraph_name),
                stmts: vec![]
            };

            // Collect all flattened nodes under the current aggregate node
            for leaf_id in leaves.iter() {
                let rir_id_opt = self.flatid(AggNodeIndex::from(agg_id), *leaf_id);
                if let Some(rir_id) = rir_id_opt {
                    let rir_node = self.graph.node_weight(*rir_id).unwrap();
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
                        if na.contains_key(rir_id) {
                            gv_node.attributes.push(na.get(rir_id).unwrap().clone());
                        }
                    }
                    subgraph.stmts.push(DotStmt::from(gv_node));
                }
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
