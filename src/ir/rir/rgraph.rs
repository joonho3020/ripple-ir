use crate::ir::rir::rnode::*;
use crate::ir::rir::redge::*;
use crate::ir::rir::IndexGen;
use crate::ir::rir::agg::*;
use crate::ir::typetree::subtree::SubTreeView;
use crate::ir::typetree::typetree::*;
use crate::ir::typetree::subtree::LeavesWithPath;
use crate::common::graphviz::*;
use chirrtl_parser::ast::*;
use petgraph::graph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::graph::EdgeIndex;
use petgraph::visit::EdgeRef;
use petgraph::{Incoming, Outgoing};
use bimap::BiMap;
use indexmap::IndexMap;
use indexmap::IndexSet;

type IRGraph = Graph<RippleNode, RippleEdge>;

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
    pub graph: IRGraph,

    /// Bi-directional map that ties a low level graph node to 
    /// each aggregate node and vice-versa
    pub agg_node_map: BiMap<RippleNodeIndex, AggNodeLeafIndex>,

    /// map that contains metadata for each unique aggregate node
    pub agg_nodes: IndexMap<AggNodeIndex, AggNodeData>,

    /// Map that ties a low level graph edge to each aggregate edge
    pub agg_edge_map: IndexMap<AggEdgeIndex, Vec<RippleEdgeIndex>>,

    /// map that contains metadata for each unique aggregate edges
    pub agg_edges: IndexMap<AggNodeIndex, Vec<AggEdge>>,

    /// Cache that maps graph nodes to its aggregate node.
    /// - Must be updated correctly when removing nodes, or just invalidated and
    /// reconstructed from scratch
    /// - This is because of how petgraph changes the `NodeIndex` when removing nodes.
    node_map_cache: BiMap<NodeIndex, AggNodeLeafIndex>,
}

impl RippleGraph {
    pub fn new() -> Self {
        Self {
            agg_node_idx_gen: IndexGen::new(),
            node_idx_gen: IndexGen::new(),

            agg_edge_idx_gen: IndexGen::new(),
            edge_idx_gen: IndexGen::new(),

            graph: IRGraph::new(),

            agg_node_map: BiMap::new(),
            agg_nodes: IndexMap::new(),

            agg_edge_map: IndexMap::new(),
            agg_edges: IndexMap::new(),

            node_map_cache: BiMap::new(),
        }
    }

    /// Adds a aggregate node into the graph.
    /// - If the node is of ground type, it will add a single node. Otherwise,
    /// the type will be flattened into separate nodes
    /// - You can still access the aggregate information using the returned
    /// `AggNodeIndex`
    pub fn add_node_agg(&mut self, node: AggNodeData) -> AggNodeIndex {
        let my_ttree = node.ttree.as_ref().unwrap().view().unwrap();
        let unique_agg_id = self.agg_node_idx_gen.generate();
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
            let unique_id = self.node_idx_gen.generate();
            let rinfo = RippleNodeData::new(Some(leaf_name), node.nt.clone(), tg.clone());
            let rgnode = RippleNode::new(rinfo, unique_id);
            let graph_node_id = self.graph.add_node(rgnode);

            // Add mapping between the aggregate node and the flattened node
            let anli = AggNodeLeafIndex::new(unique_agg_id, *leaf_id);
            self.agg_node_map.insert(unique_id, anli.clone());

            // Add mapping to cache
            self.node_map_cache.insert(graph_node_id, anli);
        }

        self.agg_nodes.insert(unique_agg_id, node);
        return unique_agg_id;
    }

    /// Given a aggregate node and its leafid in the `TypeTree`, return the
    /// NodeIndex in the graph
    pub fn flatid(&self, aggid: AggNodeIndex, leafid: TTreeNodeIndex) -> Option<&NodeIndex> {
        let aggleafidx = AggNodeLeafIndex::new(aggid, leafid);
        self.node_map_cache.get_by_right(&aggleafidx)
    }

    fn ttree_leaves_with_path(&self, id: AggNodeIndex, reference: &Reference) -> LeavesWithPath {
        let agg_node = self.agg_nodes.get(&id).unwrap();
        let ttree = agg_node.ttree.as_ref().unwrap().view().unwrap();
        ttree.subtree_leaves_with_path(reference)
    }

    fn ttree_array_entry_leaves_with_path(&self, id: AggNodeIndex, reference: &Reference) -> LeavesWithPath {
        let agg_node = self.agg_nodes.get(&id).unwrap();
        let ttree = agg_node.ttree.as_ref().unwrap().view().unwrap();
        let ttree_array_entry = ttree.subtree_array_element();
        ttree_array_entry.subtree_leaves_with_path(reference)
    }

    fn ttree_leaf(&self, id: AggNodeIndex, leaf: TTreeNodeIndex) -> Option<TypeTreeNode> {
        let agg_node = self.agg_nodes.get(&id).unwrap();
        let ttree = agg_node.ttree.as_ref().unwrap().view().unwrap();
        ttree.get_node(leaf)
    }

    fn subttree_root(&self, id: AggNodeIndex, reference: &Reference) -> Option<TTreeNodeIndex> {
        let agg_node = self.agg_nodes.get(&id).unwrap();
        let ttree = agg_node.ttree.as_ref().unwrap().view().unwrap();
        ttree.subtree_root(reference)
    }

    fn subttree_leaves(&self, id: AggNodeIndex, root: TTreeNodeIndex) -> Vec<TTreeNodeIndex> {
        let agg_node = self.agg_nodes.get(&id).unwrap();
        let ttree = agg_node.ttree.as_ref().unwrap();
        let sub_ttree = SubTreeView::new(ttree, root);
        sub_ttree.leaves()
    }

    fn ttree_leaves(&self, id: AggNodeIndex) -> Vec<TTreeNodeIndex> {
        let agg_node = self.agg_nodes.get(&id).unwrap();
        let ttree = agg_node.ttree.as_ref().unwrap().view().unwrap();
        ttree.leaves()
    }

    fn create_and_add_agg_edge(
        &mut self,
        src_id: AggNodeIndex,
        src_ref: &Reference,
        dst_id: AggNodeIndex,
        dst_ref: &Reference,
        edge: AggEdgeData
    ) -> AggEdgeIndex {
        // Create AggEdge
        let src_ttree_root = self.subttree_root(src_id, src_ref).unwrap();
        let dst_ttree_root = self.subttree_root(dst_id, dst_ref);
        let agg_edge_id = self.agg_edge_idx_gen.generate();
        let agg_edge = AggEdge::new(agg_edge_id,
            edge,
            src_id,
            dst_id,
            src_ttree_root,
            dst_ttree_root);

        // Agg AggEdge
        if !self.agg_edges.contains_key(&src_id) {
            self.agg_edges.insert(src_id, vec![]);
        }
        let agg_edges = self.agg_edges.get_mut(&src_id).unwrap();
        agg_edges.push(agg_edge);
        return agg_edge_id;
    }


    /// Adds an edge between two aggregate nodes.
    /// - Edge is flattened according to the reference used to connect these nodes
    pub fn add_edge_agg(
        &mut self,
        src_id: AggNodeIndex,
        src_ref: &Reference,
        dst_id: AggNodeIndex,
        dst_ref: &Reference,
        edge: AggEdgeData
    ) -> AggEdgeIndex {
        let src_leaves = self.ttree_leaves_with_path(src_id, src_ref);
        let dst_leaves = self.ttree_leaves_with_path(dst_id, dst_ref);

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        for (src_path_identity, src_ttree_leaf_id) in src_leaves.iter() {
            // If there is a matching path in the dst aggregate node, add an edge
            if dst_leaves.contains_key(src_path_identity) {
                // Create edge
                let unique_edge_id = self.edge_idx_gen.generate();
                let edge_info = RippleEdgeData::new(None, edge.et.clone());
                let ripple_edge = RippleEdge::new(edge_info, unique_edge_id);

                // Src node id
                let src_ttree_leaf = self.ttree_leaf(src_id, *src_ttree_leaf_id).unwrap();
                let src_flatid = self.flatid(src_id, *src_ttree_leaf_id).unwrap();

                // Dst node id
                let dst_ttree_leaf_id = dst_leaves.get(src_path_identity).unwrap();
                let dst_flatid = self.flatid(dst_id, *dst_ttree_leaf_id).unwrap();

                // Add edge
                if src_ttree_leaf.dir == TypeDirection::Outgoing {
                    edges.push((*src_flatid, *dst_flatid, ripple_edge));
                } else {
                    edges.push((*dst_flatid, *src_flatid, ripple_edge));
                }
            } else {
                eprintln!("src_ref {:?}", src_ref);
                eprintln!("src_id {:?}", src_id);
                eprintln!("src_leaves {:?}", src_leaves);
                eprintln!("dst_ref {:?}", dst_ref);
                eprintln!("dst_id {:?}", dst_id);
                eprintln!("dst_leaves {:?}", dst_leaves);
                panic!("Aggregate nodes are not connected");
            }
        }

        // Add aggregate edge
        let agg_eid = self.create_and_add_agg_edge(src_id, src_ref, dst_id, dst_ref, edge);
        if !self.agg_edge_map.contains_key(&agg_eid) {
            self.agg_edge_map.insert(agg_eid, vec![]);
        }

        for edge in edges {
            // Add to agg_edge_map
            self.agg_edge_map.get_mut(&agg_eid).unwrap().push(edge.2.id);

            // Add the flat edges
            self.graph.add_edge(edge.0, edge.1, edge.2);
        }

        return agg_eid;
    }

    /// Adds edge between two aggregate nodes.
    /// - Edge is a ground type
    /// - The edge can have many fan-outs though
    pub fn add_fanout_edge_agg(
        &mut self,
        src_id: AggNodeIndex,
        src_ref: &Reference,
        dst_id: AggNodeIndex,
        dst_ref: &Reference,
        edge: AggEdgeData
    ) -> AggEdgeIndex {
        let src_leaves = self.ttree_leaves_with_path(src_id, src_ref);
        let dst_leaves = self.ttree_leaves_with_path(dst_id, dst_ref);

        assert!(src_leaves.len() == 1, "add_single_edge got multiple src_leaves {:?}", src_leaves);
        assert!(dst_leaves.len() > 0, "add_single_edge got zero destinations");

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        let (_src_path_identity, src_ttree_leaf_id) = src_leaves.first().unwrap();
        let src_ttree_leaf = self.ttree_leaf(src_id, *src_ttree_leaf_id).unwrap();

        for (_dst_path_identity, dst_ttree_leaf_id) in dst_leaves {
            // Create edge
            let unique_edge_id = self.edge_idx_gen.generate();
            let edge_info = RippleEdgeData::new(None, edge.et.clone());
            let ripple_edge = RippleEdge::new(edge_info, unique_edge_id);

            // Find flatid
            let src_flatid = self.flatid(src_id, *src_ttree_leaf_id).unwrap();
            let dst_flatid = self.flatid(dst_id, dst_ttree_leaf_id).unwrap();

            // Add edge
            if src_ttree_leaf.dir == TypeDirection::Outgoing {
                edges.push((*src_flatid, *dst_flatid, ripple_edge));
            } else {
                edges.push((*dst_flatid, *src_flatid, ripple_edge));
            }
        }

        // Add aggregate edge
        let agg_eid = self.create_and_add_agg_edge(src_id, src_ref, dst_id, dst_ref, edge);
        if !self.agg_edge_map.contains_key(&agg_eid) {
            self.agg_edge_map.insert(agg_eid, vec![]);
        }

        for edge in edges {
            // Add to agg_edge_map
            self.agg_edge_map.get_mut(&agg_eid).unwrap().push(edge.2.id);

            // Add the flat edges
            self.graph.add_edge(edge.0, edge.1, edge.2);
        }

        return agg_eid;
    }

    /// Adds edge between two array types
    pub fn add_mem_edge_agg(
        &mut self,
        src_id: AggNodeIndex,
        src_ref: &Reference,
        dst_id: AggNodeIndex,
        dst_ref: &Reference,
        edge: AggEdgeData
    ) -> AggEdgeIndex {
        // Get leaves of the src aggregate node
        let src_leaves = self.ttree_array_entry_leaves_with_path(src_id, src_ref);
        let dst_leaves = self.ttree_leaves_with_path(dst_id, dst_ref);

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        for (src_path_key, src_ttree_leaf_id) in src_leaves.iter() {
            // If there is a matching path in the dst aggregate node, add an edge
            if dst_leaves.contains_key(src_path_key) {
                // Create edge
                let unique_edge_id = self.edge_idx_gen.generate();
                let edge_info = RippleEdgeData::new(None, edge.et.clone());
                let ripple_edge = RippleEdge::new(edge_info, unique_edge_id);

                // Src node id
                let src_ttree_leaf = self.ttree_leaf(src_id, *src_ttree_leaf_id).unwrap();
                let src_flatid = self.flatid(src_id, *src_ttree_leaf_id).unwrap();

                // Dst node id
                let dst_ttree_leaf_id = dst_leaves.get(src_path_key).unwrap();
                let dst_flatid = self.flatid(dst_id, *dst_ttree_leaf_id).unwrap();
                if src_ttree_leaf.dir == TypeDirection::Outgoing {
                    edges.push((*src_flatid, *dst_flatid, ripple_edge));
                } else {
                    edges.push((*dst_flatid, *src_flatid, ripple_edge));
                }
            } else {
                eprintln!("src_ref {:?}", src_ref);
                eprintln!("src_id {:?}", src_id);
                eprintln!("src_leaves {:?}", src_leaves);
                eprintln!("dst_ref {:?}", dst_ref);
                eprintln!("dst_id {:?}", dst_id);
                eprintln!("dst_leaves {:?}", dst_leaves);
                panic!("Aggregate nodes are not connected");
            }
        }

        // Add aggregate edge
        let agg_eid = self.create_and_add_agg_edge(src_id, src_ref, dst_id, dst_ref, edge);
        if !self.agg_edge_map.contains_key(&agg_eid) {
            self.agg_edge_map.insert(agg_eid, vec![]);
        }

        for edge in edges {
            // Add to agg_edge_map
            self.agg_edge_map.get_mut(&agg_eid).unwrap().push(edge.2.id);

            // Add the flat edges
            self.graph.add_edge(edge.0, edge.1, edge.2);
        }

        return agg_eid;
    }

    /// Similar to node_indices in petgraph.
    pub fn node_indices_agg(&self) -> Vec<AggNodeIndex> {
        (0..self.agg_nodes.len()).map(|x| AggNodeIndex::from(x)).collect()
    }

    /// Similar to node_weight in petgraph.
    pub fn node_weight_agg(&self, id: AggNodeIndex) -> Option<&AggNodeData> {
        self.agg_nodes.get(&id)
    }

    /// Returns all neighboring aggregates
    pub fn neighbors_agg(&self, agg_id: AggNodeIndex) -> Vec<AggNodeIndex> {
        let agg_node = self.agg_nodes.get(&agg_id).unwrap();
        let ttree = agg_node.ttree.as_ref().unwrap();
        let leaf_ids = ttree.view().unwrap().leaves();

        let mut neighbor_ttree_ids: IndexSet<AggNodeIndex> = IndexSet::new();
        for leaf_id in leaf_ids {
            let ir_id = *self.flatid(agg_id, leaf_id).unwrap();
            let neighbor_ids = self.graph.neighbors_directed(ir_id, Outgoing);

            for nid in neighbor_ids {
                let node = self.graph.node_weight(nid).unwrap();
                let neighbor_agg_leaf = self.agg_node_map.get_by_left(&node.id).unwrap();
                neighbor_ttree_ids.insert(neighbor_agg_leaf.agg_id);
            }

            let neighbor_ids = self.graph.neighbors_directed(ir_id, Incoming);
            for nid in neighbor_ids {
                let node = self.graph.node_weight(nid).unwrap();
                let neighbor_agg_leaf = self.agg_node_map.get_by_left(&node.id).unwrap();
                neighbor_ttree_ids.insert(neighbor_agg_leaf.agg_id);
            }
        }
        return neighbor_ttree_ids.iter().map(|x| *x).collect();
    }

    pub fn edges_agg(&self, agg_id: AggNodeIndex) -> Vec<&AggEdge> {
        self.agg_edges.get(&agg_id).unwrap().iter().collect()
    }

    pub fn flatids_under_agg(&self, agg_id: AggNodeIndex) -> Vec<NodeIndex> {
        let leaves = self.ttree_leaves(agg_id);
        leaves.iter().map(|leaf| *self.flatid(agg_id, *leaf).unwrap()).collect()
    }

    pub fn flatedges_under_agg(&self, edge: &AggEdge) -> Vec<EdgeIndex> {
        let unique_edge_ids_vec = self.agg_edge_map.get(&edge.id).unwrap();
        let unique_edge_ids: IndexSet<&RippleEdgeIndex> = IndexSet::from_iter(unique_edge_ids_vec);
        let src_ids = self.subttree_leaves(edge.src, edge.src_subtree_root);

        fn collect_edge_ids(
            rg: &RippleGraph,
            ids: &Vec<NodeIndex>,
            valid: &IndexSet<&RippleEdgeIndex>,
            dir: petgraph::Direction,
            ret: &mut Vec<EdgeIndex>,
        ) {
            for id in ids {
                let edges = rg.graph.edges_directed(*id, dir);
                for eid in edges {
                    let edge = rg.graph.edge_weight(eid.id()).unwrap();
                    if valid.contains(&edge.id) {
                        ret.push(eid.id());
                    }
                }
            }
        }

        let mut ret = vec![];
        collect_edge_ids(self, &src_ids, &unique_edge_ids, Outgoing, &mut ret);
        collect_edge_ids(self, &src_ids, &unique_edge_ids, Incoming, &mut ret);
        return ret;
    }

    pub fn vismap_agg(&self) -> AggVisMap {
        AggVisMap::new(self.agg_nodes.len() as u32)
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
        for (agg_id, agg_node) in self.agg_nodes.iter() {
            let ttree = agg_node.ttree.as_ref().unwrap();
            let leaves = ttree.view().unwrap().leaves();

            // Create graphviz subgraph to group nodes together
            let subgraph_name = format!("\"cluster_{}_{}\"",
                agg_node.name,
                agg_id.to_usize()).replace('"', "");
            let mut subgraph = DotSubgraph {
                id: Id::Plain(subgraph_name),
                stmts: vec![]
            };

            // Collect all flattened nodes under the current aggregate node
            for leaf_id in leaves.iter() {
                let rir_id_opt = self.flatid(*agg_id, *leaf_id);
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
