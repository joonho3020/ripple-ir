use std::{
    fmt::Debug,
    collections::BinaryHeap
};
use indexmap::IndexMap;
use indexmap::IndexSet;
use petgraph::{
    graph::NodeIndex,
    visit::{VisitMap, Visitable},
    Direction::Outgoing,
    Direction::Incoming,
};
use petgraph::visit::DfsPostOrder;
use crate::passes::ast::firrtlgraph::*;

/// Dice: metric representing how similar two subtrees are
#[derive(Debug, Clone, Hash, Ord, Eq)]
pub struct Dice {
    /// Number of descendents in subtree 1
    pub s1: u32,

    /// Number of descendents in subtree 2
    pub s2: u32,

    /// Number of matches under subtree 1 & subtree 2
    pub m: u32
}

impl Dice {
    pub fn new(val: f32) -> Self {
        assert!(val <= 1.0, "Dice value should be leq to 1, got {}", val);
        assert!(val > 0.0, "Dice value should be gt to 0, got {}", val);
        let m = 1000;
        let s1_plus_s2 = (2.0 / val * m as f32) as u32;
        Self {
            s1: s1_plus_s2 / 2,
            s2: s1_plus_s2 / 2,
            m,
        }
    }

    pub fn dice(&self) -> f32 {
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

#[derive(Default, Debug, Clone, Copy, Hash, Ord, Eq)]
pub struct HeightNodeIndex {
    pub height: Height,
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
/// Implements the GumTree algorithm for FIRRTL AST comparison
/// [Fine-grained and Accurate Source Code Differencing](https://hal.science/hal-01054552/document)
/// [Github](https://github.com/GumTreeDiff/gumtree)
#[derive(Debug)]
pub struct GumTree {
    /// Minium height of the tree to start finding matches
    min_height: Height,

    /// Minimum dice value to declare as a match during the bottom up phase
    min_dice: Dice,

    /// Maximum subtree size to call the O(n^3) tree-edit-distance algorithm
    _max_subtree_size: u32,
}

struct PriorityQueue(BinaryHeap<HeightNodeIndex>);

impl PriorityQueue {
    fn new() -> Self {
        Self { 0: BinaryHeap::new() }
    }

    fn peek(&self) -> Option<HeightNodeIndex> {
        self.0.peek().copied()
    }

    fn push(&mut self, hn: HeightNodeIndex) {
        self.0.push(hn)
    }

    /// Adds all the child nodes of `id` of `fg` into the priority queue
    fn open(&mut self, fg: &FirrtlGraph, id: NodeIndex) {
        for cidx in fg.graph.neighbors_directed(id, Outgoing) {
            let cnode = fg.graph.node_weight(cidx).unwrap();
            self.push(HeightNodeIndex::new(cnode.height, cidx));
        }
    }

    /// Pops all the nodes in `pq` with `height`, and adds all of their children
    /// into `pq`
    fn open_all(&mut self, fg: &FirrtlGraph, height: Height) {
        while self.peek().unwrap().height == height {
            let hnidx = self.0.pop().unwrap();
            self.open(fg, hnidx.idx);
        }
    }

    /// Removes all nodes in `self` with `height`. Returns a node hash to node index
    /// mapping
    fn pop(&mut self, fg: &FirrtlGraph, height: u32) -> HashToIndex {
        let mut hm: HashToIndex = IndexMap::new();
        while self.peek().is_some() && self.peek().unwrap().height == height {
            let hnidx = self.0.pop().unwrap();
            let node = fg.graph.node_weight(hnidx.idx).unwrap();

            // We are using this Map based approach in order to avoid
            // performing quadratic comparisons between all popped hashes.
            // Instead, we can perform quadratic comparisons only btw nodes with
            // hash matches.
            if !hm.contains_key(&node.hash) {
                hm.insert(node.hash, vec![]);
            }
            hm.get_mut(&node.hash).unwrap().push(hnidx);
        }
        return hm;
    }
}

type HashToIndex = IndexMap<HashVal, Vec<HeightNodeIndex>>;

#[derive(Debug, Clone, Copy)]
pub struct Match(pub NodeIndex, pub NodeIndex);

pub type Matches = Vec<Match>;

#[derive(Debug, Clone, Copy)]
pub struct HeightMatch(pub HeightNodeIndex, pub HeightNodeIndex);

pub type HeightMatches = Vec<HeightMatch>;

type MatchedDescSet = IndexMap<NodeIndex, Vec<NodeIndex>>;

impl Default for GumTree {
    fn default() -> Self {
        Self {
            min_height: 2,
            min_dice: Dice::new(0.5),
            _max_subtree_size: 100,
        }
    }
}

impl GumTree {
    pub fn new(min_height: u32, min_dice: Dice, max_subtree_size: u32) -> Self {
        Self {
            min_height,
            min_dice,
            _max_subtree_size: max_subtree_size,
        }
    }
}

impl GumTree {
    pub fn top_down_phase(&self, src: &FirrtlGraph, dst: &FirrtlGraph) -> Matches {
        let mut src_pq = PriorityQueue::new();
        let mut dst_pq = PriorityQueue::new();

        let mut candidates = HeightMatches::new();
        let mut mappings   = HeightMatches::new();

        // hash value to count map
        let src_hash_count = src.hash_count();
        let dst_hash_count = dst.hash_count();

        // Initialize priority queues with root nodes
        src_pq.push(HeightNodeIndex::new(src.height(src.root), src.root));
        dst_pq.push(HeightNodeIndex::new(dst.height(dst.root), dst.root));

        let mut src_vismap = src.graph.visit_map();
        let mut dst_vismap = dst.graph.visit_map();

        while src_pq.peek().unwrap().height >= self.min_height &&
              dst_pq.peek().unwrap().height >= self.min_height
        {
            let src_peek_height = src_pq.peek().unwrap().height;
            let dst_peek_height = dst_pq.peek().unwrap().height;

            if src_peek_height > dst_peek_height {
                // height mismatch, add child nodes of this height
                src_pq.open_all(src, src_peek_height);
            } else if dst_peek_height > src_peek_height {
                // height mismatch, add child nodes of this height
                dst_pq.open_all(dst, dst_peek_height);
            } else {
                let mut src_h = src_pq.pop(src, src_peek_height);
                let mut dst_h = dst_pq.pop(dst, dst_peek_height);

                // Height match, find matching hashes
                let keys2: IndexSet<_> = dst_h.keys().collect();
                let common_hashes: Vec<_> = src_h
                                            .keys()
                                            .filter(|&key| keys2.contains(key))
                                            .map(|h| h.clone())
                                            .collect();

                // iterate over the common hashes and add matching nodes to
                // either the candiate list or the mapping list
                for ch in common_hashes {
                    let src_h_nodes = src_h.swap_remove(&ch).unwrap();
                    let dst_h_nodes = dst_h.swap_remove(&ch).unwrap();

                    if *src_hash_count.get(&ch).unwrap() > 1 ||
                       *dst_hash_count.get(&ch).unwrap() > 1 {
                           for sn in src_h_nodes.iter() {
                               for dn in dst_h_nodes.iter() {
                                   candidates.push(HeightMatch(sn.clone(), dn.clone()));
                               }
                           }
                    } else {
                        assert!(src_h_nodes.len() == 1);
                        assert!(dst_h_nodes.len() == 1);

                        // Found a subtree that matches.
                        // Travserse both subtrees in the same order to add the matching
                        // nodes to the mapping
                        let mut dfs_src = DfsPostOrder::new(&src.graph, src_h_nodes[0].idx);
                        let mut dfs_dst = DfsPostOrder::new(&dst.graph, dst_h_nodes[0].idx);

                        while let (Some(nidx_src), Some(nidx_dst)) =
                            (dfs_src.next(&src.graph), dfs_dst.next(&dst.graph))
                        {
                            let node_src = src.graph.node_weight(nidx_src).unwrap();
                            let node_dst = dst.graph.node_weight(nidx_dst).unwrap();
                            mappings.push(HeightMatch(
                                HeightNodeIndex::new(node_src.height, nidx_src),
                                HeightNodeIndex::new(node_dst.height, nidx_dst)
                            ));
                            src_vismap.visit(nidx_src);
                            dst_vismap.visit(nidx_dst);
                        }
                    }
                }

                // Add child nodes that weren't matched
                for (_, hns) in src_h.iter() {
                    for hn in hns {
                        src_pq.open(src, hn.idx);
                    }
                }
                for (_, hns) in dst_h.iter() {
                    for hn in hns {
                        dst_pq.open(dst, hn.idx);
                    }
                }
            }
        }
        let mut dices = Self::dice_list(&mappings, &candidates, src, dst);
        dices.sort_by(|a, b| b.0.cmp(&a.0));

        for (_, hnidx_src, hnidx_dst) in dices {
            if !src_vismap.is_visited(&hnidx_src.idx) && !dst_vismap.is_visited(&hnidx_dst.idx) {
                let mut dfs_src = DfsPostOrder::new(&src.graph, hnidx_src.idx);
                let mut dfs_dst = DfsPostOrder::new(&dst.graph, hnidx_dst.idx);

                while let (Some(nidx_src), Some(nidx_dst)) = (dfs_src.next(&src.graph), dfs_dst.next(&dst.graph)) {
                    let node_src = src.graph.node_weight(nidx_src).unwrap();
                    let node_dst = dst.graph.node_weight(nidx_dst).unwrap();
                    mappings.push(HeightMatch(
                            HeightNodeIndex::new(node_src.height, nidx_src),
                            HeightNodeIndex::new(node_dst.height, nidx_dst)
                    ));
                    src_vismap.visit(nidx_src);
                    dst_vismap.visit(nidx_dst);
                }
            }
        }
        return mappings.iter().map(|x| Match(x.0.idx, x.1.idx)).collect();
    }

    /// Counts the number of matches under (and including) each node
    fn descendent_match_cnt(
        fg: &FirrtlGraph,
        ast_matches: &IndexSet<NodeIndex>
    ) -> IndexMap<NodeIndex, u32> {
        let mut desc_cnt: IndexMap<NodeIndex, u32> = IndexMap::new();
        let mut subtree_cnt: IndexMap<NodeIndex, u32> = IndexMap::new();

        let mut dfs = DfsPostOrder::new(&fg.graph, fg.root);
        while let Some(nidx) = dfs.next(&fg.graph) {
            let childs = fg.graph.neighbors_directed(nidx, Outgoing);
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
        mappings: &HeightMatches,
        candidates: &'a HeightMatches,
        src: &FirrtlGraph,
        dst: &FirrtlGraph
    ) -> Vec<(Dice, &'a HeightNodeIndex, &'a HeightNodeIndex)> {
        let mut dices: Vec<(Dice, &HeightNodeIndex, &HeightNodeIndex)> = vec![];

        let src_matches: IndexSet<NodeIndex> = mappings.iter().map(|x| x.0.idx).collect();
        let desc_cnt = Self::descendent_match_cnt(src, &src_matches);

        for cand in candidates {
            let nidx_src = cand.0.idx;
            let nidx_dst = cand.1.idx;

            let parents_src = src.graph.neighbors_directed(nidx_src, Incoming);
            let parents_dst = dst.graph.neighbors_directed(nidx_dst, Incoming);
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
                        s1: src.graph.node_weight(src_id).unwrap().descs,
                        s2: dst.graph.node_weight(dst_id).unwrap().descs,
                        m:  *desc_cnt.get(&src_id).unwrap_or(&0)
                    };
                    dices.push((dice, &cand.0, &cand.1));
                }
                _ => { }
            }
        }
        return dices;
    }
}

impl GumTree {
    pub fn bottom_up_phase(
        self: &Self,
        src: &FirrtlGraph,
        dst: &FirrtlGraph,
        mappings: &Matches
    ) -> Matches {
        let mut ret: Matches = Matches::new();
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

        let mut dfs_src = DfsPostOrder::new(&src.graph, src.root);
        while let Some(src_idx) = dfs_src.next(&src.graph) {
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
                            ret.push(Match(src_idx, cand));

                            // TODO: Implement tree-edit distance
                        }
                    }
                    _ => { }
                }
            }
        }
        return ret;
    }

    /// Returns a map where given a node, returns a list of descendent nodes that are matched
    fn matched_descendent_set(
        fg: &FirrtlGraph,
        matched: &IndexSet<&NodeIndex>
    ) -> MatchedDescSet {
        let mut ret: MatchedDescSet = IndexMap::new();

        let mut dfs = DfsPostOrder::new(&fg.graph, fg.root);
        while let Some(nidx) = dfs.next(&fg.graph) {
            let childs = fg.graph.neighbors_directed(nidx, Outgoing);
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
        src: &FirrtlGraph,
        dst: &FirrtlGraph,
        src_matched_desc_set: &MatchedDescSet,
        dst_matched: &IndexSet<&NodeIndex>,
        dst_matched_desc_set: &MatchedDescSet,
        src_to_dst_mapping: &IndexMap<NodeIndex, NodeIndex>,
    ) -> Option<(Dice, NodeIndex)> {
        let mut candidates: IndexMap<Dice, NodeIndex> = IndexMap::new();

        let dst_labels = dst.labels();
        let src_node = src.graph.node_weight(src_id).unwrap();
        let dst_matching_labels = dst_labels.get(&src_node.label());

        match dst_matching_labels {
            Some(dst_nodes) => {
                // for all dst graph nodes with matching labels...
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
                        let dst_node = dst.graph.node_weight(*dst_id).unwrap();
                        candidates.insert(
                            Dice {
                                s1: src_node.descs,
                                s2: dst_node.descs,
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
}

impl GumTree {
    /// Given two `LnastTree`s, returns a vector containing the tuple of matched
    /// node indices
    pub fn diff(self: &Self, src: &FirrtlGraph, dst: &FirrtlGraph) -> Matches {
        let mut td_matches = self.top_down_phase(src, dst);
        let bu_matches = self.bottom_up_phase(src, dst, &mut td_matches);
        td_matches.extend(bu_matches);

        // Make sure tree roots are matched
        let dst_matches: IndexSet<NodeIndex> = td_matches.iter().map(|x| x.1).collect();
        if !dst_matches.contains(&dst.root) {
            td_matches.push(Match(src.root, dst.root));
        }
        return td_matches;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dice_calculation() {
        let dice = Dice::new(0.5);
        assert_eq!(dice.dice(), 0.5);
        let dice = Dice {
            s1: 10,
            s2: 10,
            m: 5,
        };
        assert_eq!(dice.dice(), 0.5);
    }
}
