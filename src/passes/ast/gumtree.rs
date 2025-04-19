use std::{
    fmt::Debug,
    collections::BinaryHeap
};
use indexmap::IndexSet;
use petgraph::{
    graph::{NodeIndex, Graph},
    visit::{VisitMap, Visitable},
};
use chirrtl_parser::ast::*;
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
/// Implements the GumTree algorithm for FIRRTL AST comparison
#[derive(Debug)]
pub struct GumTree {
    min_height: u32,
    min_dice: Dice,
    max_subtree_size: u32,
}

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

    /// Compare two FIRRTL circuits using the GumTree algorithm
    pub fn compare(&self, src: &Circuit, dst: &Circuit) -> Vec<(NodeIndex, NodeIndex)> {
        let src_graph = FirrtlGraph::from_circuit(src);
        let dst_graph = FirrtlGraph::from_circuit(dst);

        // Phase 1: Top-down phase to find isomorphic subtrees
        let mut matches = self.top_down_phase(&src_graph, &dst_graph);

        // Phase 2: Bottom-up phase to find additional matches
        matches.extend(self.bottom_up_phase(&src_graph, &dst_graph, &matches));

        matches
    }

    fn top_down_phase(&self, src: &FirrtlGraph, dst: &FirrtlGraph) -> Vec<(NodeIndex, NodeIndex)> {
        let mut matches = Vec::new();
        let mut src_pq = BinaryHeap::new();
        let mut dst_pq = BinaryHeap::new();

        // Initialize priority queues with root nodes
        src_pq.push(HeightNodeIndex::new(0, src.root));
        dst_pq.push(HeightNodeIndex::new(0, dst.root));

        while !src_pq.is_empty() && !dst_pq.is_empty() {
            let src_height = src_pq.peek().unwrap().height;
            let dst_height = dst_pq.peek().unwrap().height;

            if src_height != dst_height {
                break;
            }

            // Find candidate matches at current height
            let src_candidates = self.pop_nodes_at_height(&mut src_pq, src_height);
            let dst_candidates = self.pop_nodes_at_height(&mut dst_pq, dst_height);

            // Find isomorphic subtrees
            for (src_idx, src_node) in src_candidates.iter() {
                for (dst_idx, dst_node) in &dst_candidates {
                    if self.are_nodes_isomorphic(&src.graph[*src_idx], &dst.graph[*dst_idx]) {
                        matches.push((*src_idx, *dst_idx));
                        break;
                    }
                }
            }

            // Add children to priority queues
            for (idx, _) in src_candidates {
                for neighbor in src.graph.neighbors(idx) {
                    src_pq.push(HeightNodeIndex::new(src_height + 1, neighbor));
                }
            }

            for (idx, _) in dst_candidates {
                for neighbor in dst.graph.neighbors(idx) {
                    dst_pq.push(HeightNodeIndex::new(dst_height + 1, neighbor));
                }
            }
        }
        matches
    }

    fn bottom_up_phase(
        &self,
        src: &FirrtlGraph,
        dst: &FirrtlGraph,
        matches: &[(NodeIndex, NodeIndex)]
    ) -> Vec<(NodeIndex, NodeIndex)> {
        let mut additional_matches = Vec::new();
        let matched_src: IndexSet<_> = matches.iter().map(|(s, _)| *s).collect();
        let matched_dst: IndexSet<_> = matches.iter().map(|(_, d)| *d).collect();

        // For each unmatched node in source tree
        for src_idx in src.graph.node_indices() {
            if matched_src.contains(&src_idx) {
                continue;
            }

            // Find best candidate match in destination tree
            let mut best_match = None;
            let mut best_dice = Dice::new(0.0);

            for dst_idx in dst.graph.node_indices() {
                if matched_dst.contains(&dst_idx) {
                    continue;
                }

                if self.are_nodes_similar(&src.graph[src_idx], &dst.graph[dst_idx]) {
                    let dice = self.calculate_dice(src, dst, src_idx, dst_idx, matches);
                    if dice > best_dice {
                        best_dice = dice;
                        best_match = Some(dst_idx);
                    }
                }
            }

            if let Some(dst_idx) = best_match {
                if best_dice.dice() >= self.min_dice.dice() {
                    additional_matches.push((src_idx, dst_idx));
                }
            }
        }

        additional_matches
    }

    fn pop_nodes_at_height(
        &self,
        pq: &mut BinaryHeap<HeightNodeIndex>,
        height: u32
    ) -> Vec<(NodeIndex, HeightNodeIndex)> {
        let mut nodes = Vec::new();
        while let Some(node) = pq.peek() {
            if node.height != height {
                break;
            }
            let node = pq.pop().unwrap();
            nodes.push((node.idx, node));
        }
        nodes
    }

    fn are_nodes_isomorphic(&self, a: &FirrtlNode, b: &FirrtlNode) -> bool {
        match (a, b) {
            (FirrtlNode::Circuit(a), FirrtlNode::Circuit(b)) => a.name == b.name,
            (FirrtlNode::Module(a), FirrtlNode::Module(b)) => a.name == b.name,
            (FirrtlNode::ExtModule(a), FirrtlNode::ExtModule(b)) => a.name == b.name,
            (FirrtlNode::Port(a), FirrtlNode::Port(b)) => a == b,
            (FirrtlNode::Stmt(a), FirrtlNode::Stmt(b)) => a == b,
            (FirrtlNode::Type(a), FirrtlNode::Type(b)) => a == b,
            (FirrtlNode::Expr(a), FirrtlNode::Expr(b)) => a == b,
            (FirrtlNode::Reference(a), FirrtlNode::Reference(b)) => a == b,
            (FirrtlNode::Info(a), FirrtlNode::Info(b)) => a == b,
            _ => false,
        }
    }

    fn are_nodes_similar(&self, a: &FirrtlNode, b: &FirrtlNode) -> bool {
        std::mem::discriminant(a) == std::mem::discriminant(b)
    }

    fn calculate_dice(
        &self,
        src: &FirrtlGraph,
        dst: &FirrtlGraph,
        src_idx: NodeIndex,
        dst_idx: NodeIndex,
        matches: &[(NodeIndex, NodeIndex)]
    ) -> Dice {
        let mut common_descendants = 0;
        let src_descendants = self.count_descendants(&src.graph, src_idx);
        let dst_descendants = self.count_descendants(&dst.graph, dst_idx);

        for (s, d) in matches {
            if self.is_descendant(&src.graph, src_idx, *s) &&
               self.is_descendant(&dst.graph, dst_idx, *d) {
                common_descendants += 1;
            }
        }

        Dice {
            s1: src_descendants,
            s2: dst_descendants,
            m: common_descendants,
        }
    }

    fn count_descendants(&self, graph: &Graph<FirrtlNode, ()>, node: NodeIndex) -> u32 {
        let mut count = 0;
        let mut visited = graph.visit_map();
        let mut stack = vec![node];

        while let Some(current) = stack.pop() {
            if !visited.visit(current) {
                count += 1;
                stack.extend(graph.neighbors(current));
            }
        }

        count - 1 // Exclude the node itself
    }

    fn is_descendant(&self, graph: &Graph<FirrtlNode, ()>, ancestor: NodeIndex, descendant: NodeIndex) -> bool {
        let mut visited = graph.visit_map();
        let mut stack = vec![ancestor];

        while let Some(current) = stack.pop() {
            if current == descendant {
                return true;
            }
            if !visited.visit(current) {
                stack.extend(graph.neighbors(current));
            }
        }

        false
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
