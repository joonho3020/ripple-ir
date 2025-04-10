use super::whentree::*;
use chirrtl_parser::ast::*;
use derivative::Derivative;
use petgraph::{graph::{Graph, NodeIndex}, visit::Bfs, Direction::Outgoing};
use petgraph::Direction::Incoming;
use indexmap::IndexMap;
use indexmap::IndexSet;
use std::{collections::VecDeque, u32, usize};
use std::cmp::max as higher;
use std::hash::Hash;

use crate::define_index_type;

impl WhenTree {
    fn print_tree_recursive(&self, id: NodeIndex, depth: usize) {
        println!("{}{:?} {:?}", "  ".repeat(depth), self.graph.node_weight(id).unwrap(), id);

        let childs: Vec<NodeIndex> = self.graph.neighbors_directed(id, Outgoing)
                                                .into_iter()
                                                .collect();
        for child in childs.iter().rev() {
            self.print_tree_recursive(*child, depth + 1);
        }
    }

    /// Print the tree to stdout
    pub fn print_tree(&self) {
        match self.god {
            Some(rid) => {
                self.print_tree_recursive(rid, 0);
            }
            _ => {
                println!("WhenTree empty");
            }
        }
    }
}

impl WhenTree {
    /// Performs some basic checks about the `WhenTree`
    pub fn check_validity(&self) {
        let mut unique_priorities: IndexSet<PhiPrior> = IndexSet::new();

        let mut bfs = Bfs::new(&self.graph, self.god.unwrap());
        while let Some(id) = bfs.next(&self.graph) {
            let num_childs = self.graph.neighbors_directed(id, Outgoing).count();
            if num_childs == 0 {
                let node = self.graph.node_weight(id).unwrap();

                // Check for uniqueness of leaf node's priority
                assert!(!unique_priorities.contains(&node.prior));
                unique_priorities.insert(node.prior);

                // Leaf node should be of a root type
                assert!(node.cond == Condition::Root);
            }
        }
    }
}

impl WhenTree {
    /// Recursively build the tree from given `stmts`
    fn build_from_stmts_recursive(
        &mut self,
        prior: &mut PhiPrior,
        parent_id: NodeIndex,
        stmts: &Stmts,
    ) {
        let mut cur_node: Option<&mut WhenTreeNode> = None;

        // Traverse in reverse order to take last-connection semantics into account
        for (stmt_prior, stmt) in stmts.iter().rev().enumerate() {
            // Update priority stmt
            prior.stmt = StmtPrior(stmt_prior as u32);

            match stmt.as_ref() {
                Stmt::When(cond, _info, when_stmts, else_stmts_opt) => {
                    cur_node = None;

                    // Update priority
                    prior.inc_block();

                    // Add new when node to the tree
                    let when_cond = Condition::When(cond.clone());
                    let id = self.graph.add_node(WhenTreeNode::new(when_cond, *prior));
                    self.graph.add_edge(parent_id, id, WhenTreeEdge::default());

                    // Build when subtree
                    self.build_from_stmts_recursive(
                        prior,
                        id,
                        when_stmts);

                    prior.inc_block();

                    if let Some(else_stmts) = else_stmts_opt {
                        // When and Else nodes share the same stmt priority
                        prior.stmt = StmtPrior(stmt_prior as u32);

                        // Add new else node to the tree
                        let else_cond = Condition::Else(cond.clone());
                        let id = self.graph.add_node(WhenTreeNode::new(else_cond, *prior));
                        self.graph.add_edge(parent_id, id, WhenTreeEdge::default());

                        // Build else subtree
                        self.build_from_stmts_recursive(
                            prior,
                            id,
                            else_stmts);

                        prior.inc_block();
                    }
                }
                _ => {
                    match cur_node.as_mut() {
                        None => {
                            // Add node to parent
                            let tn = WhenTreeNode::new(Condition::Root, *prior);
                            let child_id = self.graph.add_node(tn);
                            self.graph.add_edge(parent_id, child_id, WhenTreeEdge::default());
                            cur_node = Some(self.graph.node_weight_mut(child_id).unwrap());
                        }
                        _ => {}
                    }
                    let pstmt = PrioritizedStmt::new(stmt.as_ref().clone(), prior.stmt);
                    cur_node.as_mut().unwrap().stmts.push(pstmt);
                }
            }
        }
    }

    /// Creates a when tree from given `Stmts`
    pub fn build_from_stmts(stmts: &Stmts) -> Self {
        let mut ret = Self::default();
        let god_node = WhenTreeNode::god();
        let god_id = ret.graph.add_node(god_node);

        ret.god = Some(god_id);
        ret.build_from_stmts_recursive(&mut PhiPrior::bottom(), god_id, stmts);
        ret.check_validity();
        ret
    }
}

impl WhenTree {
    /// Returns all the leaf nodes along with the condition path to reach it
    pub fn leaf_to_conditions(&self) -> LeafToPath {
        let mut ret = LeafToPath::new();
        let mut q: VecDeque<(NodeIndex, PrioritizedCondPath)> = VecDeque::new();
        q.push_back((self.god.unwrap(), PrioritizedCondPath::default()));

        // BFS
        while !q.is_empty() {
            let nid_path = q.pop_front().unwrap();
            let nid = nid_path.0;
            let cur_path = &nid_path.1;

            let mut num_childs = 0;
            for cid in self.graph.neighbors_directed(nid, Outgoing) {
                let cnode = self.graph.node_weight(cid).unwrap();
                num_childs += 1;

                // Extend path with the current cond & prior
                let mut path = cur_path.clone();
                let pcond = PrioritizedCond::new(cnode.prior, cnode.cond.clone());
                path.push(pcond);

                // Add child to BFS queue
                q.push_back((cid, path));
            }

            let node = self.graph.node_weight(nid).unwrap();
            if num_childs == 0 {
                // Leaf node
                ret.insert(node, cur_path.clone());
            } else {
                // Non leaf node, should not contain statement
                assert!(node.stmts.is_empty());
            }
        }
        return ret;
    }
}

impl WhenTree {
    fn add_node_and_connect(
        tree: &mut WhenTree,
        node: WhenTreeNode,
        parent_id: NodeIndex
    ) -> NodeIndex {
        let id = tree.graph.add_node(node);
        tree.graph.add_edge(parent_id, id, WhenTreeEdge::default());
        id
    }

    /// Reconstructs a WhenTree from a vector of `PrioritizedCondPath`
    pub fn from_conditions(cond_paths: Vec<&PrioritizedCondPath>) -> Self {
        let mut tree = Self::default();

        // Set up the god node
        let god_node = WhenTreeNode::god();
        let god_id = tree.graph.add_node(god_node);
        tree.god = Some(god_id);

        // Mapping of already inserted conditions to node ids (for non-leaf nodes)
        let mut cond_to_node: IndexMap<&PrioritizedCond, NodeIndex> = IndexMap::new();

        // Root conditions (leaf nodes) can be indexed by its unique priority
        let mut visited_leaf_prior: IndexSet<PhiPrior> = IndexSet::new();

        // Sort by priority
        let mut sorted_paths = cond_paths;
        sorted_paths.sort_by(|a, b| b.cmp(a));

        // Iterate over found condition paths
        for pcond_path in sorted_paths {
            let mut cur_id = god_id;

            // Traverse down the tree attaching nodes as needed
            for pcond in pcond_path.iter() {
                // Leaf node
                if pcond.cond == Condition::Root {
                    if visited_leaf_prior.contains(&pcond.prior) {
                        // Already visited this node
                        continue;
                    } else {
                        // Found a Root condition: add as leaf node if not visited
                        let node = WhenTreeNode::new(pcond.cond.clone(), pcond.prior);
                        Self::add_node_and_connect(&mut tree, node, cur_id);
                        visited_leaf_prior.insert(pcond.prior);
                    }
                } else if !cond_to_node.contains_key(pcond) {
                    // Condition not yet visited: add to tree
                    let node = WhenTreeNode::new(pcond.cond.clone(), pcond.prior);
                    let id = Self::add_node_and_connect(&mut tree, node, cur_id);
                    cond_to_node.insert(pcond, id);
                    cur_id = id;
                } else {
                    // Already added this condition before.
                    // Go down the tree by following this map
                    cur_id = *cond_to_node.get(pcond).unwrap();
                }
            }
        }

        // Add a node that will be the default node where stmts under to condition
        // will be inserted into if it already doesn't exist
        let mut has_bottom_node = false;
        for cid in tree.graph.neighbors_directed(god_id, Outgoing) {
            let child = tree.graph.node_weight(cid).unwrap();
            if child.prior == PhiPrior::bottom() && child.cond == Condition::Root {
                has_bottom_node = true;
                break;
            }
        }

        // Add bottom node if it doesn't exist
        if !has_bottom_node {
            let id = tree.graph.add_node(WhenTreeNode::new(Condition::Root, PhiPrior::bottom()));
            tree.graph.add_edge(god_id, id, WhenTreeEdge::default());
        }

        // Get priority of highest located when and root nodes
        let mut highest_when_prior = PhiPrior::bottom();
        let mut highest_root_prior = PhiPrior::bottom();
        for cid in tree.graph.neighbors_directed(god_id, Outgoing) {
            let child = tree.graph.node_weight(cid).unwrap();
            match child.cond {
                Condition::When(..) => {
                    highest_when_prior = higher(highest_when_prior, child.prior);
                }
                Condition::Root => {
                    highest_root_prior = higher(highest_root_prior, child.prior);
                }
                _ => {
                }
            }
        }

        // Add top node if needed
        if highest_when_prior > highest_root_prior {
            let id = tree.graph.add_node(WhenTreeNode::new(Condition::Root, PhiPrior::top()));
            tree.graph.add_edge(god_id, id, WhenTreeEdge::default());
        }

        tree
    }

    /// Get the PrioritizedConds that represents the first stmt block (on the top)
    pub fn get_top_pcond(&self) -> PrioritizedCondPath {
        let mut highest_root_prior = PhiPrior::bottom();
        for cid in self.graph.neighbors_directed(self.god.unwrap(), Outgoing) {
            let child = self.graph.node_weight(cid).unwrap();
            match child.cond {
                Condition::Root => {
                    highest_root_prior = higher(highest_root_prior, child.prior);
                }
                _ => {
                }
            }
        }
        PrioritizedCondPath(vec![
            PrioritizedCond::new(highest_root_prior, Condition::Root)
        ])
    }
}
