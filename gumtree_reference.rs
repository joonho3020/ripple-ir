use crate::{
    livehd::LiveHDError,
    LnastTree,
    LnastHash,
    LnastTreeNode,
    LiveHDBuilder,
};
use std::{
    fmt::Debug,
    collections::{BinaryHeap, VecDeque}
};
use graphviz_rust::{
    attributes::{rankdir, color_name, EdgeAttributes, GraphAttributes, NodeAttributes},
    exec_dot,
    dot_structures::*,
    dot_generator::{edge, node_id, id},
    cmd::{CommandArg, Format},
    printer::{DotPrinter, PrinterContext},
};
use indexmap::{IndexMap, IndexSet};
use petgraph::{
    graph::{NodeIndex, Graph}, Direction::{Outgoing, Incoming},
    visit::{DfsPostOrder, VisitMap, Visitable},
};

/// Dice: metric representing how similar two subtrees are
#[derive(Debug, Clone, Hash, Ord, Eq)]
pub struct Dice {
    /// Number of descendents in subtree 1
    pub s1: u32,

    /// Number of descendents in subtree 2
    pub s2: u32,

    /// Number of matches under subtree 1 & subtree 2
    pub m:  u32
}

impl Dice {
    pub fn new(val: f32) -> Self {
        assert!(val <= 1.0, "Dice value should be leq to 1, got {}", val);
        assert!(val >  0.0, "Dice value should be gt to 0, got {}", val);
        let m = 1000;
        let s1_plus_s2 = (2.0 / val * m as f32) as u32;
        Self {
            s1: s1_plus_s2 / 2,
            s2: s1_plus_s2 / 2,
            m,
        }
    }

    pub fn dice(self: &Self) -> f32 {
        2.0 * self.m as f32 / (self.s1 + self.s2) as f32
    }
}

impl PartialEq for Dice {
    fn eq(&self, other: &Self) -> bool {
        (self.s1 + self.s2) * other.m == (other.s1 + other.s2) * self.m
    }
}

impl PartialOrd for Dice {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let my_dice = self.m * (other.s1 + other.s2);
        let other_dice = other.m * (self.s1 + self.s2);
        if my_dice > other_dice {
            Some(std::cmp::Ordering::Greater)
        } else if my_dice == other_dice {
            Some(std::cmp::Ordering::Equal)
        } else {
            Some(std::cmp::Ordering::Less)
        }
    }
}

#[derive(Default, Debug, Clone, Hash, Ord, Eq)]
struct HeightNodeIndex {
    pub height: u32,
    pub idx: NodeIndex,
}

impl HeightNodeIndex {
    fn new(height: u32, idx: NodeIndex) -> Self {
        Self {
            height,
            idx,
        }
    }
}

impl PartialEq for HeightNodeIndex {
    fn eq(&self, other: &Self) -> bool {
        self.height == other.height
    }
}

impl PartialOrd for HeightNodeIndex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.height > other.height {
            Some(std::cmp::Ordering::Greater)
        } else if self.height == other.height {
            Some(std::cmp::Ordering::Equal)
        } else {
            Some(std::cmp::Ordering::Less)
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PrintGraphEdge {
    SrcEdge,
    DstEdge,
    TopDownEdge,
    BottomUpEdge,
}

impl PrintGraphEdge {
    fn color(self: &Self) -> color_name {
        match self {
            Self::SrcEdge => { color_name::black },
            Self::DstEdge => { color_name::black },
            Self::TopDownEdge => { color_name::red },
            Self::BottomUpEdge => { color_name::blue },
        }
    }
}

/// Implements the gumtree algorithm
/// [Fine-grained and Accurate Source Code Differencing](https://hal.science/hal-01054552/document)
/// [Github](https://github.com/GumTreeDiff/gumtree)
#[derive(Debug, Clone)]
pub struct GumTree {
    /// Minium height of the tree to start finding matches
    pub min_height: u32,

    /// Minimum dice value to declare as a match during the bottom up phase
    pub min_dice: Dice,

    /// Maximum subtree size to call the O(n^3) tree-edit-distance algorithm
    pub max_subtree_size: u32,
}

type PriorityQueue = BinaryHeap<HeightNodeIndex>;
type HashToIndex = IndexMap<LnastHash, Vec<HeightNodeIndex>>;
type HeightASTMatch = (HeightNodeIndex, HeightNodeIndex);
type MatchedDescSet = IndexMap<NodeIndex, Vec<NodeIndex>>;
type TreeNodeIndex = NodeIndex;
type PrintNodeIndex = NodeIndex;
type PrintGraph = Graph<LnastTreeNode, PrintGraphEdge>;

pub type ASTMatch = (NodeIndex, NodeIndex);

impl Default for GumTree {
    fn default() -> Self {
        Self {
            min_height: 2,
            min_dice: Dice::new(0.5),
            max_subtree_size: 100,
        }
    }
}

impl GumTree {
    pub fn new(min_height: u32, min_dice: Dice, max_subtree_size: u32) -> Self {
        Self {
            min_height,
            min_dice,
            max_subtree_size,
        }
    }

    /// Adds all the child nodes of `nidx` in `ln` into `pq`
    fn pq_open(ln: &LnastTree, pq: &mut PriorityQueue, nidx: &NodeIndex) {
        let childs = ln.tree.neighbors_directed(*nidx, Outgoing);
        for cidx in childs {
            let cnode = ln.tree.node_weight(cidx).unwrap();
            pq.push(HeightNodeIndex::new(cnode.height, cidx));
        }
    }

    /// Pops all the nodes in `pq` with `height`, and adds all of their children
    /// into `pq`
    fn pq_open_all(ln: &LnastTree, pq: &mut PriorityQueue, height: u32) {
        while pq.peek().unwrap().height == height {
            let hnidx = pq.pop().unwrap();
            Self::pq_open(ln, pq, &hnidx.idx);
        }
    }

    /// Removes all nodes in `pq` with `height`. Returns a node hash to node index
    /// mapping
    fn pq_pop(ln: &LnastTree, pq: &mut PriorityQueue, height: u32) -> HashToIndex {
        let mut hm: HashToIndex = IndexMap::new();
        while pq.peek().is_some() && pq.peek().unwrap().height == height {
            let hnidx = pq.pop().unwrap();
            let node = ln.tree.node_weight(hnidx.idx).unwrap();

            // We are using this Map based approach in order to avoid
            // performing quadratic comparisons between all popped hashes.
            // Instead, we can perform quadratic comparisons only btw nodes with
            // hash matches.
            if !hm.contains_key(&node.hash.unwrap()) {
                hm.insert(node.hash.unwrap(), vec![]);
            }
            hm.get_mut(&node.hash.unwrap()).unwrap().push(hnidx);
        }
        return hm;
    }

    fn descendent_match_cnt(
        ln: &LnastTree,
        ast_matches: &IndexSet<NodeIndex>
    ) -> IndexMap<NodeIndex, u32> {
        let mut desc_cnt: IndexMap<NodeIndex, u32> = IndexMap::new();
        let mut subtree_cnt: IndexMap<NodeIndex, u32> = IndexMap::new();

        let mut dfs = DfsPostOrder::new(&ln.tree, ln.root);
        while let Some(nidx) = dfs.next(&ln.tree) {
            let childs = ln.tree.neighbors_directed(nidx, Outgoing);
            let mut num_match_cnt = 0;
            for cidx in childs {
                num_match_cnt += *subtree_cnt.get(&cidx).unwrap_or(&0);
            }
            desc_cnt.insert(nidx, num_match_cnt);

            if ast_matches.contains(&nidx) {
                subtree_cnt.insert(nidx, num_match_cnt + 1);
            } else {
                subtree_cnt.insert(nidx, num_match_cnt);
            }
        }
        return desc_cnt;
    }

    fn dice_list<'a>(
        mappings: &Vec<HeightASTMatch>,
        candidates: &'a Vec<HeightASTMatch>,
        src: &LnastTree,
        dst: &LnastTree
    ) -> Vec<(Dice, &'a HeightNodeIndex, &'a HeightNodeIndex)> {
        let mut dices: Vec<(Dice, &HeightNodeIndex, &HeightNodeIndex)> = vec![];

        let src_matches: IndexSet<NodeIndex> = mappings.iter().map(|x| x.0.idx).collect();
        let desc_cnt = Self::descendent_match_cnt(src, &src_matches);

        for cand in candidates {
            let nidx_src = cand.0.idx;
            let nidx_dst = cand.1.idx;

            let parents_src = src.tree.neighbors_directed(nidx_src, Incoming);
            let parents_dst = dst.tree.neighbors_directed(nidx_dst, Incoming);
            let mut pidx_src: Option<NodeIndex> = None;
            let mut pidx_dst: Option<NodeIndex> = None;
            for p in parents_src {
                pidx_src = Some(p);
            }
            for p in parents_dst {
                pidx_dst = Some(p);
            }
            match (pidx_src, pidx_dst) {
                (Some(src_id), Some(dst_id)) => {
                    let dice = Dice {
                        s1: src.tree.node_weight(src_id).unwrap().num_desc,
                        s2: dst.tree.node_weight(dst_id).unwrap().num_desc,
                        m:  *desc_cnt.get(&src_id).unwrap_or(&0)
                    };
                    dices.push((dice, &cand.0, &cand.1));
                }
                _ => { }
            }
        }
        return dices;
    }

    fn top_down_phase(self: &Self, src: &LnastTree, dst: &LnastTree) -> Vec<ASTMatch> {
        let mut pq_src: PriorityQueue = BinaryHeap::new();
        let mut pq_dst: PriorityQueue = BinaryHeap::new();

        let src_hash_count = src.hash_count();
        let dst_hash_count = dst.hash_count();

        let src_root = src.tree.node_weight(src.root).unwrap();
        let dst_root = dst.tree.node_weight(dst.root).unwrap();

        let mut candidates: Vec<HeightASTMatch> = vec![];
        let mut mappings:   Vec<HeightASTMatch> = vec![];

        pq_src.push(HeightNodeIndex::new(src_root.height, src.root));
        pq_dst.push(HeightNodeIndex::new(dst_root.height, dst.root));

        let mut vismap_src = src.tree.visit_map();
        let mut vismap_dst = dst.tree.visit_map();

        while pq_src.peek().unwrap().height >= self.min_height &&
              pq_dst.peek().unwrap().height >= self.min_height
        {
            let pq_src_peek_height = pq_src.peek().unwrap().height;
            let pq_dst_peek_height = pq_dst.peek().unwrap().height;

            if pq_src_peek_height > pq_dst_peek_height {
                // height mismatch, add child nodes of this height
                Self::pq_open_all(src, &mut pq_src, pq_src_peek_height);
            } else if pq_dst_peek_height > pq_src_peek_height {
                // height mismatch, add child nodes of this height
                Self::pq_open_all(dst, &mut pq_dst, pq_dst_peek_height);
            } else {
                let mut h_src = Self::pq_pop(src, &mut pq_src, pq_src_peek_height);
                let mut h_dst = Self::pq_pop(dst, &mut pq_dst, pq_dst_peek_height);

                // height match, find matching hashes
                let keys2: IndexSet<_> = h_dst.keys().collect();
                let common_hashes: Vec<_> = h_src
                                            .keys()
                                            .filter(|&key| keys2.contains(key))
                                            .map(|h| h.clone())
                                            .collect();

                // iterate over the common hashes and add matching nodes to
                // either the candiate list or the mapping list
                for ch in common_hashes {
                    let h_src_nodes = h_src.swap_remove(&ch).unwrap();
                    let h_dst_nodes = h_dst.swap_remove(&ch).unwrap();

                    if *src_hash_count.get(&ch).unwrap() > 1 ||
                       *dst_hash_count.get(&ch).unwrap() > 1 {
                           for sn in h_src_nodes.iter() {
                               for dn in h_dst_nodes.iter() {
                                   candidates.push((sn.clone(), dn.clone()));
                               }
                           }
                    } else {
                        assert!(h_src_nodes.len() == 1);
                        assert!(h_dst_nodes.len() == 1);

                        // Found a subtree that matches.
                        // Travserse both subtrees in the same order to add the matching
                        // nodes to the mapping
                        let mut dfs_src = DfsPostOrder::new(&src.tree, h_src_nodes[0].idx);
                        let mut dfs_dst = DfsPostOrder::new(&dst.tree, h_dst_nodes[0].idx);

                        while let (Some(nidx_src), Some(nidx_dst)) =
                            (dfs_src.next(&src.tree), dfs_dst.next(&dst.tree))
                        {
                            let node_src = src.tree.node_weight(nidx_src).unwrap();
                            let node_dst = dst.tree.node_weight(nidx_dst).unwrap();
                            mappings.push((
                                HeightNodeIndex::new(node_src.height, nidx_src),
                                HeightNodeIndex::new(node_dst.height, nidx_dst)
                            ));
                            vismap_src.visit(nidx_src);
                            vismap_dst.visit(nidx_dst);
                        }
                    }
                }

                // Add child nodes that weren't matched
                for (_, hns) in h_src.iter() {
                    for hn in hns {
                        Self::pq_open(src, &mut pq_src, &hn.idx);
                    }
                }
                for (_, hns) in h_dst.iter() {
                    for hn in hns {
                        Self::pq_open(dst, &mut pq_dst, &hn.idx);
                    }
                }
            }
        }

        let mut dices = Self::dice_list(&mappings, &candidates, src, dst);
        dices.sort_by(|a, b| b.0.cmp(&a.0));

        for (_, hnidx_src, hnidx_dst) in dices {
            if !vismap_src.is_visited(&hnidx_src.idx) && !vismap_dst.is_visited(&hnidx_dst.idx) {
                let mut dfs_src = DfsPostOrder::new(&src.tree, hnidx_src.idx);
                let mut dfs_dst = DfsPostOrder::new(&dst.tree, hnidx_dst.idx);

                while let (Some(nidx_src), Some(nidx_dst)) = (dfs_src.next(&src.tree), dfs_dst.next(&dst.tree)) {
                    let node_src = src.tree.node_weight(nidx_src).unwrap();
                    let node_dst = dst.tree.node_weight(nidx_dst).unwrap();
                    mappings.push((
                            HeightNodeIndex::new(node_src.height, nidx_src),
                            HeightNodeIndex::new(node_dst.height, nidx_dst)
                    ));
                    vismap_src.visit(nidx_src);
                    vismap_dst.visit(nidx_dst);
                }
            }
        }
        return mappings.iter().map(|x| (x.0.idx, x.1.idx)).collect();
    }


    /// Returns a map where given a node, returns a list of descendent nodes that are matched
    fn matched_descendent_set(ln: &LnastTree, matched: &IndexSet<&NodeIndex>) -> MatchedDescSet {
        let mut ret: MatchedDescSet = IndexMap::new();

        let mut dfs = DfsPostOrder::new(&ln.tree, ln.root);
        while let Some(nidx) = dfs.next(&ln.tree) {
            let childs = ln.tree.neighbors_directed(nidx, Outgoing);
            let mut desc_matches: Vec<NodeIndex> = vec![];
            for child in childs {
                desc_matches.extend(ret.get(&child).unwrap().iter());
                if matched.contains(&child) {
                    desc_matches.push(child);
                }
            }
            ret.insert(nidx, desc_matches);
        }
        return ret;
    }

    fn bottom_up_candidate(
        src_id: NodeIndex,
        src: &LnastTree,
        dst: &LnastTree,
        src_matched_desc_set: &MatchedDescSet,
        dst_matched: &IndexSet<&NodeIndex>,
        dst_matched_desc_set: &MatchedDescSet,
        src_to_dst_mapping: &IndexMap<NodeIndex, NodeIndex>,
    ) -> Option<(Dice, NodeIndex)> {
        let mut candidates: IndexMap<Dice, NodeIndex> = IndexMap::new();

        let dst_labels = dst.labels();
        let src_node = src.tree.node_weight(src_id).unwrap();
        let dst_matching_labels = dst_labels.get(&src_node.label());

        match dst_matching_labels {
            Some(dst_nodes) => {
                // for all dst tree nodes with matching labels...
                for dst_id in dst_nodes {
                    // if the dst node is not matched...
                    if !dst_matched.contains(&dst_id) {
                        let mut matched_descendent_cnt = 0;

                        // descendent nodes of the dst node that is matched
                        let dst_matched_descs: IndexSet<&NodeIndex> = dst_matched_desc_set
                                                                        .get(dst_id)
                                                                        .unwrap()
                                                                        .iter()
                                                                        .collect();
                        let src_matched_descs = src_matched_desc_set.get(&src_id).unwrap();
                        for smd in src_matched_descs {
                            let dst_matched_id = src_to_dst_mapping.get(smd).unwrap();
                            if dst_matched_descs.contains(dst_matched_id) {
                                matched_descendent_cnt += 1;
                            }
                        }
                        let dst_node = dst.tree.node_weight(*dst_id).unwrap();
                        candidates.insert(
                            Dice {
                                s1: src_node.num_desc,
                                s2: dst_node.num_desc,
                                m: matched_descendent_cnt,
                            },
                            *dst_id,
                        );
                    }
                }
            }
            None => { }
        }

        let max_key_opt = candidates.keys().max();
        match max_key_opt {
            Some(max_key) => {
                Some((max_key.clone(), *candidates.get(max_key).unwrap()))
            }
            _ => {
                None
            }
        }
    }

    fn bottom_up_phase(self: &Self, src: &LnastTree, dst: &LnastTree, mappings: &Vec<ASTMatch>) -> Vec<ASTMatch> {
        let mut ret: Vec<ASTMatch> = vec![];
        let src_matched: IndexSet<&NodeIndex> = mappings.iter().map(|x| &x.0).collect();
        let dst_matched: IndexSet<&NodeIndex> = mappings.iter().map(|x| &x.1).collect();
        let src_matched_desc_set = Self::matched_descendent_set(src, &src_matched);
        let dst_matched_desc_set = Self::matched_descendent_set(dst, &dst_matched);

        let src_to_dst_mapping: IndexMap<NodeIndex, NodeIndex> = mappings
                                                                    .iter()
                                                                    .map(|x| (x.0, x.1))
                                                                    .collect();
        let src_matched_desc_cnt: IndexMap<NodeIndex, u32> = src_matched_desc_set
                                                                .iter()
                                                                .map(|(k, v)| (*k, v.len() as u32))
                                                                .collect();
        let mut dfs_src = DfsPostOrder::new(&src.tree, src.root);
        while let Some(src_idx) = dfs_src.next(&src.tree) {
            if !src_matched.contains(&src_idx) &&
               *src_matched_desc_cnt.get(&src_idx).unwrap() > 0
            {
                let candidate_opt = Self::bottom_up_candidate(
                    src_idx,
                    src,
                    dst,
                    &src_matched_desc_set,
                    &dst_matched,
                    &dst_matched_desc_set,
                    &src_to_dst_mapping);

                match candidate_opt {
                    Some((dice, cand)) => {
                        if dice > self.min_dice {
                            ret.push((src_idx, cand));

                            // TODO: Implement tree-edit distance
                        }
                    }
                    _ => { }
                }
            }
        }
        return ret;
    }

    /// Given two `LnastTree`s, returns a vector containing the tuple of matched
    /// node indices
    pub fn diff(self: &Self, src: &LnastTree, dst: &LnastTree) -> Vec<ASTMatch> {
        if src.hash().unwrap() == dst.hash().unwrap() {
            return vec![];
        } else {
            let mut td_matches = self.top_down_phase(src, dst);
            let bu_matches = self.bottom_up_phase(src, dst, &mut td_matches);
            td_matches.extend(bu_matches);

            // Make sure tree roots are matched
            let dst_matches: IndexSet<NodeIndex> = td_matches.iter().map(|x| x.1).collect();
            if !dst_matches.contains(&dst.root) {
                td_matches.push((src.root, dst.root));
            }
            return td_matches;
        }
    }

    fn add_graph(
        ln: &LnastTree,
        print_graph: &mut Graph::<LnastTreeNode, PrintGraphEdge>,
        edge_type: PrintGraphEdge,
    ) -> IndexMap<TreeNodeIndex, PrintNodeIndex> {
        let mut src_map: IndexMap<TreeNodeIndex, PrintNodeIndex> = IndexMap::new();
        for idx in ln.tree.node_indices() {
            let w = ln.tree.node_weight(idx).unwrap();
            let print_idx = print_graph.add_node(w.clone());
            src_map.insert(idx, print_idx);
        }
        for idx in ln.tree.edge_indices() {
            let ep = ln.tree.edge_endpoints(idx).unwrap();
            print_graph.add_edge(
                *src_map.get(&ep.0).unwrap(),
                *src_map.get(&ep.1).unwrap(),
                edge_type);
        }
        return src_map;
    }

    fn graphviz(
        pg: &PrintGraph,
        src_nodes: &Vec<NodeIndex>,
        dst_nodes: &Vec<NodeIndex>
    ) -> graphviz_rust::dot_structures::Graph {
        let mut g = graphviz_rust::dot_structures::Graph::DiGraph {
            id: graphviz_rust::dot_generator::id!(""),
            strict: true,
            stmts: vec![
                Stmt::from(GraphAttributes::rankdir(rankdir::TB))
            ]
        };

        // add invisible nodes to constrain placement
        let helper1 = "Helper1".to_string();
        let helper2 = "Helper2".to_string();

        g.add_stmt(Stmt::Node(Node {
            id: NodeId(Id::Plain(helper1.clone()), None),
            attributes: vec![
                NodeAttributes::style("invis".to_string())
            ],
        }));

        g.add_stmt(Stmt::Node(Node {
            id: NodeId(Id::Plain(helper2.clone()), None),
            attributes: vec![
                NodeAttributes::style("invis".to_string())
            ],
        }));

        let mut helper_edge = edge!(node_id!(helper1.clone()) => node_id!(helper2.clone()));
        helper_edge.attributes.push(EdgeAttributes::style("invis".to_string()));
        g.add_stmt(Stmt::Edge(helper_edge));

        // cluster of source AST
        let mut src_sg = graphviz_rust::dot_structures::Subgraph {
            id: Id::Plain("cluster_src_tree".to_string()),
            stmts: vec![]
        };

        for id in src_nodes.iter() {
            let node = pg.node_weight(*id).unwrap();

            let mut gv_node = Node {
                id: NodeId(Id::Plain(id.index().to_string()), None),
                attributes: vec![
                    NodeAttributes::label(format!("\"{}\"", node).to_string())
                ],
            };

            // Check if the node has been deleted from the source AST
            let mut has_match = false;
            for eref in pg.edges_directed(*id, Outgoing) {
                match eref.weight() {
                    PrintGraphEdge::TopDownEdge |
                        PrintGraphEdge::BottomUpEdge => {
                        has_match = true;
                    }
                    _ => { }
                }
            }

            if !has_match {
                gv_node.attributes.push(NodeAttributes::color(color_name::red));
            }

            src_sg.stmts.push(Stmt::Node(gv_node));

            // add constraints that will enforce rankings between the
            // source graph nodes and the helper node
            let mut helper_edge = edge!(
                node_id!(id.index().to_string()) =>
                node_id!(helper1.clone()));
            helper_edge.attributes.push(EdgeAttributes::style("invis".to_string()));
            g.add_stmt(Stmt::Edge(helper_edge));
        }

        g.add_stmt(Stmt::Subgraph(src_sg));

        // cluster of dst AST
        let mut dst_sg = graphviz_rust::dot_structures::Subgraph {
            id: Id::Plain("cluster_dst_tree".to_string()),
            stmts: vec![]
        };

        for id in dst_nodes.iter() {
            let node = pg.node_weight(*id).unwrap();
            let mut gv_node = Node {
                id: NodeId(Id::Plain(id.index().to_string()), None),
                attributes: vec![
                    NodeAttributes::label(format!("\"{}\"", node).to_string())
                ],
            };

            // Check if the node has been added to the dst AST
            let mut has_match = false;
            for eref in pg.edges_directed(*id, Incoming) {
                match eref.weight() {
                    PrintGraphEdge::TopDownEdge |
                        PrintGraphEdge::BottomUpEdge => {
                        has_match = true;
                    }
                    _ => { }
                }
            }

            if !has_match {
                gv_node.attributes.push(NodeAttributes::color(color_name::green));
            }

            dst_sg.stmts.push(Stmt::Node(gv_node));

            // add constraints that will enforce rankings between the
            // helper node and the dst graph nodes
            let mut helper_edge = edge!(
                node_id!(helper2.clone()) =>
                node_id!(id.index().to_string()));
            helper_edge.attributes.push(EdgeAttributes::style("invis".to_string()));
            g.add_stmt(Stmt::Edge(helper_edge));
        }

        g.add_stmt(Stmt::Subgraph(dst_sg));

        // add edges
        for eidx in pg.edge_indices() {
            let ep = pg.edge_endpoints(eidx).unwrap();
            let w = pg.edge_weight(eidx).unwrap();

            let mut e = edge!(node_id!(ep.0.index().to_string()) =>
                              node_id!(ep.1.index().to_string()));
            e.attributes.push(EdgeAttributes::color(w.color()));
            match w {
                PrintGraphEdge::TopDownEdge |
                    PrintGraphEdge::BottomUpEdge => {
                    e.attributes.push(EdgeAttributes::constraint(false));
                }
                _ => {
                }
            }
            g.add_stmt(Stmt::Edge(e));
        }
        return g;
    }

    pub fn export(
        self: &Self,
        src: &LnastTree,
        dst: &LnastTree,
        path: &str
    ) -> Result<(), LiveHDError> {
        let mut print_graph: PrintGraph = Graph::new();
        let src_map = Self::add_graph(src, &mut print_graph, PrintGraphEdge::SrcEdge);
        let dst_map = Self::add_graph(dst, &mut print_graph, PrintGraphEdge::DstEdge);

        let td_matches = self.top_down_phase(src, dst);
        for (src_id, dst_id) in td_matches.iter() {
            let p_src_id = src_map.get(src_id).unwrap();
            let p_dst_id = dst_map.get(dst_id).unwrap();
            print_graph.add_edge(*p_src_id, *p_dst_id, PrintGraphEdge::TopDownEdge);
        }

        let bu_matches = self.bottom_up_phase(src, dst, &td_matches);
        for (src_id, dst_id) in bu_matches.iter() {
            let p_src_id = src_map.get(src_id).unwrap();
            let p_dst_id = dst_map.get(dst_id).unwrap();
            print_graph.add_edge(*p_src_id, *p_dst_id, PrintGraphEdge::BottomUpEdge);
        }

        let src_print_nodes: Vec<NodeIndex> = src_map.iter().map(|(_, v)| *v).collect();
        let dst_print_nodes: Vec<NodeIndex> = dst_map.iter().map(|(_, v)| *v).collect();

        let g = Self::graphviz(&print_graph, &src_print_nodes, &dst_print_nodes);
        let dot = g.print(&mut PrinterContext::default());
        exec_dot(dot, vec![Format::Pdf.into(), CommandArg::Output(path.to_string())])?;

        Ok(())
    }
}

fn lnast_trees(pb_file: &str, top: &str) -> Vec<LnastTree> {
    let r_builder = LiveHDBuilder::new(pb_file, top);
    let lir = r_builder.build();
    return lir.iter().map(|x| x.lnast.clone()).collect();
}

fn run_gumtree_test(
    testname: &str,
    src: &mut LnastTree,
    dst: &mut LnastTree,
    odir: &str
) -> Result<(), LiveHDError> {
    // Run the gumtree algorithm
    let gumtree = GumTree::default();
    let matches = gumtree.diff(&src, &dst);

    let src_matches: IndexSet<NodeIndex> = matches.iter().map(|(a, _)| *a).collect();
    let dst_matches: IndexSet<NodeIndex> = matches.iter().map(|(_, b)| *b).collect();
    let dst2src: IndexMap<NodeIndex, NodeIndex> = matches.iter().map(|x| (x.1, x.0)).collect();

    let match_cnt = matches.len() as u32;
    let src_tree_size = src.tree.node_count() as u32;
    let dst_tree_size = dst.tree.node_count() as u32;

    println!("Src tree {} Dst tree {} Match {} ({}%)",
        src_tree_size,
        dst_tree_size,
        match_cnt,
        match_cnt as f32 / src_tree_size as f32 * 100.0);

    let mut deletions: Vec<NodeIndex> = src.tree
        .node_indices()
        .filter(|x| !src_matches.contains(x))
        .collect();

    let additions: Vec<NodeIndex> = dst.tree
        .node_indices()
        .filter(|x| !dst_matches.contains(x))
        .collect();

    type OldIndex = NodeIndex;
    type NewIndex = NodeIndex;
    let mut add_map: IndexMap<OldIndex, NewIndex> = IndexMap::new();

    type ChildIndex = NodeIndex;
    type ParentIndex = NodeIndex;
    let mut add_par: IndexMap<ChildIndex, ParentIndex> = IndexMap::new();

    type DstTreeIndex = NodeIndex;
    type SrcTreeIndex = NodeIndex;
    let mut inserted_mappings: IndexMap<DstTreeIndex, SrcTreeIndex> = IndexMap::new();

    for id in additions.iter() {
        let added = dst.tree.node_weight(*id).unwrap();
        let new_id = src.tree.add_node(added.clone());
        add_map.insert(*id, new_id);

        let parents = dst.tree.neighbors_directed(*id, Incoming);
        for pid in parents {
            add_par.insert(*id, pid);
        }
        inserted_mappings.insert(*id, new_id);
    }

    for id in additions.iter() {
        // Add parent edges
        let pid = add_par.get(id).unwrap();
        if add_map.contains_key(pid) {
            // parent is also an added node
            src.tree.add_edge(
                *add_map.get(pid).unwrap(),
                *add_map.get(id).unwrap(),
                ());
        } else {
            // parent is a matched node
            src.tree.add_edge(
                *dst2src.get(pid).unwrap(),
                *add_map.get(id).unwrap(),
                ());
        }

        // Add child edges
        let childs = dst.tree.neighbors_directed(*id, Outgoing);
        for cid in childs {
            if !add_map.contains_key(&cid) {
                src.tree.add_edge(
                    *add_map.get(id).unwrap(),
                    *dst2src.get(&cid).unwrap(),
                    ());
            }
        }
    }

    let delete_nodes: IndexSet<NodeIndex> = deletions
                                                .iter()
                                                .map(|x| *x)
                                                .collect();

    let mut dst_q: VecDeque<NodeIndex> = VecDeque::new();
    let mut src_q: VecDeque<NodeIndex> = VecDeque::new();

    dst_q.push_back(dst.root);
    src_q.push_back(src.root);

    let mut has_mismatch = false;
    'bfs: while !dst_q.is_empty() {
        let src_id = src_q.pop_front().unwrap();
        let dst_id = dst_q.pop_front().unwrap();
        let src_node = src.tree.node_weight(src_id).unwrap();
        let dst_node = dst.tree.node_weight(dst_id).unwrap();

// println!("=== BFS ===");
// println!("src: {:?}, dst: {:?}", src_node, dst_node);

        if src_node.nt != dst_node.nt ||
           src_node.token.text != dst_node.token.text {
               has_mismatch = true;
               break;
        } else {
            let src_childs = src.tree.neighbors_directed(src_id, Outgoing);
            let src_child_idxs: IndexSet<NodeIndex> =
                src_childs
                    .clone()
                    .into_iter()
                    .filter(|id| !delete_nodes.contains(id))
                    .collect();

            let dst_childs = dst.tree.neighbors_directed(dst_id, Outgoing);
            for dc in dst_childs {
                dst_q.push_back(dc);

                if inserted_mappings.contains_key(&dc) {
                    let src_id = inserted_mappings.get(&dc).unwrap();
                    src_q.push_back(*src_id);

                    assert!(src_child_idxs.contains(src_id));
                    assert!(!delete_nodes.contains(src_id));
                } else if dst2src.contains_key(&dc) {
                    let src_id = dst2src.get(&dc).unwrap();
                    src_q.push_back(*src_id);

                    assert!(!delete_nodes.contains(src_id));
                } else {
                    eprintln!("dst node not found anywhere");
                    has_mismatch = true;
                    break 'bfs;
                }
            }
        }
    }

    // Sort in decreasing order of nodes.
    // This is because how petgraph handles deletions by copying the last element
    // into the deleted index.
    // Also, we must perform deletions after the traversal has happened as it messes
    // up the existing node mappings.
    deletions.sort_by(|a, b| b.cmp(a));
    for delete_idx in deletions {
        src.tree.remove_node(delete_idx);
    }

    if src.tree.node_count() != dst.tree.node_count() ||
        src.tree.edge_count() != dst.tree.edge_count() {
        has_mismatch = true;

        eprintln!("Src V: {} E: {} Dst V: {} E: {}",
            src.tree.node_count(),
            src.tree.edge_count(),
            dst.tree.node_count(),
            dst.tree.edge_count());
    }

    if !has_mismatch {
        return Ok(());
    } else {
        src.export_tree(&format!("{}/{}.gumtree.src.txt", odir, testname))?;
        dst.export_tree(&format!("{}/{}.gumtree.dst.txt", odir, testname))?;
        return Err(LiveHDError::GumTreeReconstructFailed);
    }
}

/// Given two `LnastTree`s, runs the gumtree diffing algorithm to find the matchings.
/// Based on the matchings, reconstruct the destination AST from the source AST and
/// check for equivalence.
pub fn run(src_pb: &str, dst_pb: &str, src_top: &str, dst_top: &str, odir: &str) -> bool {
    let lnast_trees_1: Vec<LnastTree> = lnast_trees(src_pb, src_top);
    let lnast_trees_2: Vec<LnastTree> = lnast_trees(dst_pb, dst_top);
    for (src, dst) in lnast_trees_1.iter().zip(lnast_trees_2.iter()) {
        match run_gumtree_test(src_top, &mut src.clone(), &mut dst.clone(), odir) {
            Ok(_) => { }
            _ => { return false; }
        }
    }
    return true;
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use test_case::test_case;

    #[test_case("./examples/GCD.ch.pb", "./examples/GCD1.ch.pb", "GCD", "GCD1" ; "GCD")]
    #[test_case("./examples/PipelinedAdder.ch.pb", "./examples/PipelinedALU.ch.pb", "PipelinedAdder", "PipelinedALU" ; "PipelinedAdder")]
    #[test_case("./examples/MyQueue.ch.pb", "./examples/MyQueue1.ch.pb", "MyQueue", "MyQueue1" ; "Queue")]
    #[test_case("./examples/MyQueue.ch.pb", "./examples/MyQueue2.ch.pb", "MyQueue", "MyQueue2" ; "Queue1")]
    #[test_case("./examples/MyQueue2.ch.pb", "./examples/MyQueue.ch.pb", "MyQueue2", "MyQueue" ; "Queue2")]
    #[test_case("./examples/MyQueue1.ch.pb", "./examples/MyQueue2.ch.pb", "MyQueue1", "MyQueue2" ; "Queue3")]
    pub fn test(src_pb: &str, dst_pb: &str, src_top: &str, dst_top: &str) {
        assert_eq!(run(src_pb, dst_pb, src_top, dst_top, "test-outputs"), true);
    }
}
