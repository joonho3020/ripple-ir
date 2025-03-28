use crate::ir::IndexGen;
use crate::ir::rir::rgraph::*;
use crate::ir::rir::rnode::*;
use crate::ir::rir::redge::*;
use crate::ir::rir::agg::*;
use crate::ir::typetree::subtree::SubTreeView;
use crate::ir::typetree::tnode::*;
use crate::ir::typetree::subtree::LeavesWithPath;
use chirrtl_parser::ast::*;
use petgraph::graph::NodeIndex;
use indexmap::IndexSet;
use petgraph::graph::EdgeIndex;
use petgraph::visit::EdgeRef;
use petgraph::{Incoming, Outgoing};

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
            agg_neighbors: IndexMap::new(),

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
            let anli = AggNodeLeafIndex::new(unique_agg_id, leaf.id.unwrap());
            self.agg_node_map.insert(unique_id, anli.clone());

            // Add mapping to cache
            self.node_map_cache.insert(graph_node_id, anli);
        }

        self.agg_nodes.insert(unique_agg_id, node);
        return unique_agg_id;
    }

    /// Given a aggregate node and its leafid in the `TypeTree`, return the
    /// NodeIndex in the graph
    fn flatid(&self, aggid: AggNodeIndex, leafid: TypeTreeNodeIndex) -> Option<&NodeIndex> {
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

    fn ttree_leaf(&self, id: AggNodeIndex, leaf: TypeTreeNodeIndex) -> Option<TypeTreeNode> {
        let agg_node = self.agg_nodes.get(&id).unwrap();
        let ttree = agg_node.ttree.as_ref().unwrap().view().unwrap();
        ttree.get_node(leaf)
    }

    fn subttree_root(&self, id: AggNodeIndex, reference: &Reference) -> Option<TypeTreeNodeIndex> {
        let agg_node = self.agg_nodes.get(&id).unwrap();
        let ttree = agg_node.ttree.as_ref().unwrap().view().unwrap();
        ttree.subtree_root(reference)
    }

    fn subttree_all_ids(&self, id: AggNodeIndex, root: TypeTreeNodeIndex) -> Vec<TypeTreeNodeIndex> {
        let agg_node = self.agg_nodes.get(&id).unwrap();
        let ttree = agg_node.ttree.as_ref().unwrap();
        let root_graph = ttree.graph_id(root).unwrap();
        let sub_ttree = SubTreeView::new(ttree, *root_graph);
        sub_ttree.all_ids()
    }

    fn ttree_all_ids(&self, id: AggNodeIndex) -> Vec<TypeTreeNodeIndex> {
        let agg_node = self.agg_nodes.get(&id).unwrap();
        let ttree = agg_node.ttree.as_ref().unwrap().view().unwrap();
        ttree.all_ids()
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
        if !self.agg_neighbors.contains_key(&src_id) {
            self.agg_neighbors.insert(src_id, vec![]);
        }
        let agg_neighbors = self.agg_neighbors.get_mut(&src_id).unwrap();
        agg_neighbors.push(agg_edge);
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
        match self.agg_neighbors.get(&agg_id) {
            Some(neighbors) => {
                neighbors.iter().collect()
            }
            None => { vec![] }
        }
    }

    pub fn flatids_under_agg(&self, agg_id: AggNodeIndex) -> Vec<NodeIndex> {
        let ids = self.ttree_all_ids(agg_id);
        ids.iter()
            .map(|id| self.flatid(agg_id, *id))
            .filter(|x| x.is_some())
            .map(|x| *x.unwrap())
            .collect()
    }

    fn collect_edge_ids(
        &self,
        ids: &Vec<&NodeIndex>,
        valid: &IndexSet<&RippleEdgeIndex>,
        dir: petgraph::Direction,
        ret: &mut Vec<EdgeIndex>,
    ) {
        for id in ids {
            let edges = self.graph.edges_directed(**id, dir);
            for eid in edges {
                let edge = self.graph.edge_weight(eid.id()).unwrap();
                if valid.contains(&edge.id) {
                    ret.push(eid.id());
                }
            }
        }
    }

    pub fn flatedges_under_agg(&self, edge: &AggEdge) -> Vec<EdgeIndex> {
        let unique_edge_ids_vec = self.agg_edge_map.get(&edge.id).unwrap();
        let unique_edge_ids: IndexSet<&RippleEdgeIndex> = IndexSet::from_iter(unique_edge_ids_vec);
        let src_subtree_leaves = self.subttree_all_ids(edge.src, edge.src_subtree_root);
        let src_ids = src_subtree_leaves
            .iter()
            .map(|leaf_id| self.flatid(edge.src, *leaf_id))
            .filter(|x| x.is_some())
            .map(|x| x.unwrap())
            .collect();

        let mut ret = vec![];
        self.collect_edge_ids(&src_ids, &unique_edge_ids, Outgoing, &mut ret);
        self.collect_edge_ids(&src_ids, &unique_edge_ids, Incoming, &mut ret);
        return ret;
    }

    pub fn flatedges_dir_under_agg(&self, edge: &AggEdge, dir: petgraph::Direction) -> Vec<EdgeIndex> {
        let unique_edge_ids_vec = self.agg_edge_map.get(&edge.id).unwrap();
        let unique_edge_ids: IndexSet<&RippleEdgeIndex> = IndexSet::from_iter(unique_edge_ids_vec);
        let src_subtree_leaves = self.subttree_all_ids(edge.src, edge.src_subtree_root);
        let src_ids = src_subtree_leaves
            .iter()
            .map(|leaf_id| self.flatid(edge.src, *leaf_id))
            .filter(|x| x.is_some())
            .map(|x| x.unwrap())
            .collect();

        let mut ret = vec![];
        self.collect_edge_ids(&src_ids, &unique_edge_ids, dir, &mut ret);
        return ret;
    }

    pub fn vismap_agg(&self) -> AggVisMap {
        AggVisMap::new(self.agg_nodes.len() as u32)
    }

    pub fn add_node(&mut self, node: RippleNodeData) -> (RippleNodeIndex, NodeIndex) {
        let unique_id = self.node_idx_gen.generate();
        let node_with_id = RippleNode::new(node, unique_id);
        let id = self.graph.add_node(node_with_id);
        return (unique_id, id);
    }

    pub fn add_edge(&mut self, src: NodeIndex, dst: NodeIndex, edge: RippleEdgeData) -> (RippleEdgeIndex, EdgeIndex) {
        let unique_id = self.edge_idx_gen.generate();
        let edge_with_id = RippleEdge::new(edge, unique_id);
        let id = self.graph.add_edge(src, dst, edge_with_id);
        return (unique_id, id);
    }

    pub fn edge_endpoints(&self, id: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.graph.edge_endpoints(id)
    }

    pub fn edge_weight(&self, id: EdgeIndex) -> Option<&RippleEdge> {
        self.graph.edge_weight(id)
    }

    fn update_node_map_cache(&mut self) {
        self.node_map_cache.clear();
        for id in self.graph.node_indices() {
            let rid = self.graph.node_weight(id).unwrap().id;
            let aggleafidx = self.agg_node_map.get_by_left(&rid).unwrap();
            self.node_map_cache.insert(id, *aggleafidx);
        }
    }

    /// Given an aggregate node, replace all the underlying flat nodes
    /// into a single flat node
    pub fn merge_nodes_array_agg(&mut self, id: AggNodeIndex, node: RippleNodeData) {
        // Add new node to the graph
        let (uid, gid) = self.add_node(node);

        // Connect the new node to its existing edges
        let agg_edges: Vec<AggEdge> = self.edges_agg(id).iter().map(|x| (**x).clone()).collect();
        for agg_edge in agg_edges.iter() {
            let edges = self.flatedges_dir_under_agg(agg_edge, Outgoing);

            // Remove old RippleEdgeIndex's from agg_edge_map
            self.agg_edge_map.get_mut(&agg_edge.id).unwrap().clear();

            for eid in edges {
                let dst = self.edge_endpoints(eid).unwrap().1;
                let ew = self.edge_weight(eid).unwrap();
                let (ueid, _) = self.add_edge(gid, dst, ew.data.clone());

                // Add new RippleEdgeIndex's into agg_edge_map
                self.agg_edge_map.get_mut(&agg_edge.id).unwrap().push(ueid);
            }
        }

        // Get unique node index of array element in typetree
        let agg_node = self.node_weight_agg(id).unwrap();
        let ttree = agg_node.ttree.as_ref().unwrap().view().unwrap();
        let ttree_array_entry = ttree.subtree_array_element();
        let ttree_node_id = ttree_array_entry.root_node().unwrap().id.unwrap();

        // - Remove old RippleNodeIndex from agg_node_map
        // - Remove nodes from the graph
        let mut ids_under_agg = self.flatids_under_agg(id);
        ids_under_agg.sort();
        for nid in ids_under_agg.iter().rev() {
            let agnli = self.node_map_cache.get_by_left(&nid).unwrap();
            self.agg_node_map.remove_by_right(agnli);
            self.graph.remove_node(*nid);
        }

        // Add new RippleNodeIndex into agg_node_map
        self.agg_node_map.insert(uid, AggNodeLeafIndex::new(id ,ttree_node_id));

        // - Update node_map_cache
        self.update_node_map_cache();
    }
}
