use super::whentree::*;
use chirrtl_parser::ast::*;
use petgraph::{graph::NodeIndex, visit::Bfs, Direction::Outgoing};
use petgraph::Direction::Incoming;
use indexmap::IndexMap;
use indexmap::IndexSet;
use std::{collections::VecDeque, u32, usize};
use std::cmp::max as higher;

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
                    let pstmt = StmtWithPrior::new(stmt.as_ref().clone(), Some(prior.stmt));
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
    pub fn build_from_conditions(cond_paths: Vec<&CondPath>) -> Self {
        let mut tree = Self::default();

        // Set up the god node
        let god_node = WhenTreeNode::god();
        let god_id = tree.graph.add_node(god_node);
        tree.god = Some(god_id);

        // Mapping of already inserted conditions to node ids (for non-leaf nodes)
        let mut cond_to_node: IndexMap<&CondWithPrior, NodeIndex> = IndexMap::new();

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
        tree.check_validity();
        tree
    }

    /// Get the PrioritizedConds that represents the first stmt block (on the top)
    pub fn get_top_path(&self) -> CondPath {
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
        CondPath(vec![
            CondWithPrior::new(highest_root_prior, Condition::Root)
        ])
    }
}

impl WhenTree {
    /// Returns all the leaf nodes along with the condition path to reach it
    pub fn leaf_to_paths(&self) -> LeafToPath {
        let mut ret = LeafToPath::new();
        let mut q: VecDeque<(NodeIndex, CondPath)> = VecDeque::new();
        q.push_back((self.god.unwrap(), CondPath::default()));

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
                let pcond = CondWithPrior::from(cnode);
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

    /// Returns the node index in the tree following conds
    fn node_with_path(&self, path: &CondPath) -> Option<NodeIndex> {
        let mut q: VecDeque<NodeIndex> = VecDeque::new();
        q.push_back(self.god.unwrap());

        let mut i = 0;
        while !q.is_empty() && i < path.len() {
            let id = q.pop_front().unwrap();
            for cid in self.graph.neighbors_directed(id, Outgoing) {
                let pcond = path.get(i);
                let child = self.graph.node_weight(cid).unwrap();

                // Found root node
                if pcond.cond == Condition::Root &&
                    child.cond == Condition::Root &&
                    pcond.prior == child.prior
                {
                    return Some(cid);
                } else if child.cond == pcond.cond && child.prior == pcond.prior {
                    if i == path.len() - 1 {
                        return Some(cid);
                    } else {
                        // Found a matching condition, go down one level in the tree
                        q.push_back(cid);
                        i += 1;
                    }
                }
            }
        }
        None
    }

    /// Returns the prioritized cond of this node
    fn node_path(&self, id: NodeIndex) -> CondPath {
        let mut ancestors = Self::collect_ancestors(&self.graph, id);

        // remove god node
        ancestors.pop();

        // reverse order
        ancestors.reverse();

        // collect PrioritizedConds
        let mut ret = CondPath::default();
        for id in ancestors {
            let node = self.graph.node_weight(id).unwrap();
            let pcond = CondWithPrior::from(node);
            ret.push(pcond);
        }
        return ret;
    }


    /// Find all ancestors of a node (including itself), walking up toward the root
    fn collect_ancestors(graph: &WhenTreeGraph, start: NodeIndex) -> Vec<NodeIndex> {
        let mut ancestors = vec![start];
        let mut current = start;
        while let Some(parent) = graph.neighbors_directed(current, Incoming).next() {
            ancestors.push(parent);
            current = parent;
        }
        ancestors
    }

    /// Finds the LCA for the nodes corresponding to the given `paths`
    fn lowest_common_ancester(&self, paths: &Vec<CondPath>) -> Option<NodeIndex> {
        if paths.is_empty() {
            return None;
        }
        let nodes: Vec<NodeIndex> = paths.iter().map(|c| self.node_with_path(c).unwrap()).collect();

        // Collect ancestor chains for each node
        let ancestor_chains: Vec<Vec<NodeIndex>> =
            nodes.iter().map(|&n| Self::collect_ancestors(&self.graph, n)).collect();

        // Reverse chains so they go from root -> leaf
        let ancestor_chains: Vec<Vec<NodeIndex>> =
            ancestor_chains.into_iter().map(|mut v| { v.reverse(); v }).collect();

        // Find the longest common prefix across all chains
        let mut lca = None;
        let min_len = ancestor_chains.iter().map(|v| v.len()).min().unwrap_or(0);

        for i in 0..min_len {
            let candidate = ancestor_chains[0][i];
            if ancestor_chains.iter().all(|chain| chain[i] == candidate) {
                lca = Some(candidate);
            } else {
                break;
            }
        }
        lca
    }

    /// Find bottom up priority placement
    /// - Returns the CondPath that is located at the lowest (closest to bottom) position
    /// that is a root node while also being a LCA of the `conds`
    pub fn bottom_up_priority_constraint(&self, conds: &Vec<CondPath>) -> Option<CondPath> {
        let lca_opt = self.lowest_common_ancester(conds);

        if let Some(lca) = lca_opt {
            let node = self.graph.node_weight(lca).unwrap();
            if node.cond == Condition::Root && node.prior != PhiPrior::god() {
                Some(self.node_path(lca))
            } else {
                let childs = self.graph.neighbors_directed(lca, Outgoing);
                let mut cur_prior = PhiPrior::bottom();
                let mut id: Option<NodeIndex> = None;
                for cid in childs {
                    let child = self.graph.node_weight(cid).unwrap();
                    if child.cond != Condition::Root {
                        continue;
                    }
                    if id.is_none() {
                        cur_prior = child.prior;
                    } else if cur_prior == higher(cur_prior, child.prior) {
                        cur_prior = child.prior;
                        id = Some(cid);
                    }
                }

                if id.is_some() {
                    Some(self.node_path(id.unwrap()))
                } else {
                    None
                }
            }
        } else {
            None
        }
    }

    /// Find the highest root node that covers both highest and lowest
    pub fn find_middle_ground(&self, highest: &CondPath, lowest: &CondPath) -> Option<CondPath> {
        let lca_opt = self.lowest_common_ancester(&vec![highest.clone(), lowest.clone()]);
        if let Some(lca) = lca_opt {
            let childs = self.graph.neighbors_directed(lca, Outgoing);
            let mut cur_path: Option<CondPath> = None;
            for cid in childs {
                let child = self.graph.node_weight(cid).unwrap();
                if child.cond != Condition::Root {
                    continue;
                }
                let cpath = self.node_path(cid);
                if highest == &cpath {
                    cur_path = Some(cpath);
                } else if highest > &cpath && &cpath > lowest || &cpath == lowest {
                    if cur_path.is_none() {
                        cur_path = Some(cpath);
                    } else {
                        cur_path = Some(higher(cur_path.unwrap(), cpath));
                    }
                }
            }
            cur_path
        } else {
            None
        }
    }

    /// Checks if all conditions in the tree are covered given a vector of `PrioritizedCondPath`
    pub fn all_cases_covered(&self, conds: &Vec<CondPath>) -> bool {
        let lca_opt = self.lowest_common_ancester(conds);
        match lca_opt {
            Some(lca) => {
                if lca == self.god.unwrap() {
                    true
                } else {
                    false
                }
            }
            None => {
                false
            }
        }
    }
}

impl WhenTree {

    /// Follow a given condition (and the priority when it is given) to the tree leaf node
    /// and return a mutable reference to it
    pub fn get_node_mut(
        &mut self,
        conds: &CondPath,
        prior: Option<&PhiPrior>
    ) -> Option<&mut WhenTreeNode> {
        let mut q: VecDeque<NodeIndex> = VecDeque::new();
        q.push_back(self.god.unwrap());

        let mut i = 0;
        while !q.is_empty() && i < conds.len() {
            let id = q.pop_front().unwrap();
            for cid in self.graph.neighbors_directed(id, Outgoing) {
                let pcond = conds.get(i);
                let child = self.graph.node_weight(cid).unwrap();

                // Found root node
                if pcond.cond == Condition::Root &&
                    child.cond == Condition::Root &&
                    pcond.prior == child.prior
                {
                    if prior.is_some() {
                        // Priority given, check if it matches
                        if &child.prior == prior.unwrap() {
                            return self.graph.node_weight_mut(cid);
                        }
                    } else {
                        // No priority given, just return the first matching one
                        return self.graph.node_weight_mut(cid);
                    }
                } else if child.cond == pcond.cond &&
                    child.prior == pcond.prior {
                    // Found a matching condition, go down one level in the tree
                    q.push_back(cid);
                    i += 1;
                }
            }
        }
        None
    }

    fn to_stmts_recursive(&self, id: NodeIndex, stmts: &mut Stmts) {
        // Can be cases when there are multiple When Stmts with the same conditional Expr
        type WhenElseNodePair = (Option<NodeIndex>, Option<NodeIndex>);
        let mut cond_groups: IndexMap<Expr, Vec<WhenElseNodePair>> = IndexMap::new();
        let mut raw_stmt_nodes: Vec<(CondPath, NodeIndex)> = vec![];

        let childs: Vec<NodeIndex> = self.graph.neighbors_directed(id, Outgoing)
                                                .into_iter()
                                                .collect();

        // Collect children grouped by condition expression
        for &cid in childs.iter().rev() {
            let child = self.graph.node_weight(cid).unwrap();
            match &child.cond {
                Condition::Root => {
                    let pconds = CondPath::from_vec(vec![
                        CondWithPrior::new(child.prior, Condition::Root)
                    ]);
                    raw_stmt_nodes.push((pconds, cid));
                }
                Condition::When(expr) => {
                    if !cond_groups.contains_key(expr) {
                        cond_groups.insert(expr.clone(), vec![]);
                    }
                    let we_pairs = cond_groups.get_mut(expr).unwrap();
                    if we_pairs.last().is_none() || we_pairs.last().unwrap().0.is_some() {
                        we_pairs.push(WhenElseNodePair::default());
                    }
                    we_pairs.last_mut().unwrap().0 = Some(cid);
                }
                Condition::Else(expr) => {
                    if !cond_groups.contains_key(expr) {
                        cond_groups.insert(expr.clone(), vec![]);
                    }
                    let we_pairs = cond_groups.get_mut(expr).unwrap();

                    // In case the the when just contained a skip stmt
                    if we_pairs.last().is_none() || we_pairs.last().unwrap().0.is_none() {
                        we_pairs.push(WhenElseNodePair::default());
                    }
                    we_pairs.last_mut().unwrap().1 = Some(cid);
                }
            }
        }

        let mut when_stmts_by_priority: Vec<(CondPath, Stmt)> = vec![];

        // Recurse on condition branches and collect them by priority
        for (expr, we_vec) in cond_groups {
            for (when_id_opt, else_id_opt) in we_vec {
                let mut when_stmts = Stmts::new();
                if let Some(when_id) = when_id_opt {
                    self.to_stmts_recursive(when_id, &mut when_stmts);
                } else {
                    when_stmts.push(Box::new(Stmt::Skip(Info::default())));
                };

                let else_stmts_opt =  if let Some(else_id) = else_id_opt {
                    let mut else_stmts = Stmts::new();
                    self.to_stmts_recursive(else_id, &mut else_stmts);
                    Some(else_stmts)
                } else {
                    None
                };

                let stmt = Stmt::When(expr.clone(), Info::default(), when_stmts, else_stmts_opt);
                let when_prior = if let Some(when_id) = when_id_opt {
                    self.graph.node_weight(when_id).unwrap().prior
                } else {
                    // No when stmt: stmts inside when is only a skip
                    self.graph.node_weight(else_id_opt.unwrap()).unwrap().prior
                };
                let pconds = CondPath::from_vec(vec![
                    CondWithPrior::new(when_prior, Condition::When(expr.clone()))
                ]);
                when_stmts_by_priority.push((pconds, stmt));
            }
        }

        enum RawOrWhen {
            When(Stmt),
            Raw(NodeIndex)
        }

        // Merge raw stmts and when stmts and sort them by priority
        let mut merged: Vec<(CondPath, RawOrWhen)> = vec![];
        merged.extend(when_stmts_by_priority.into_iter().map(|(p, s)| (p, RawOrWhen::When(s))));
        merged.extend(raw_stmt_nodes.into_iter().map(|(p, id)| (p, RawOrWhen::Raw(id))));
        merged.sort_by(|a, b| b.0.cmp(&a.0));

        // Iterate over the merged stmts and push them in
        for (_p, raw_or_when) in merged {
            match raw_or_when {
                RawOrWhen::When(stmt) => {
                    stmts.push(Box::new(stmt));
                }
                RawOrWhen::Raw(id) => {
                    let node = self.graph.node_weight(id).unwrap();
                    for stmt in node.stmts.iter() {
                        stmts.push(Box::new(stmt.stmt.clone()));
                    }
                }
            }
        }
    }

    /// Reconstructs FIRRTL `Stmts` from the WhenTree.
    pub fn to_stmts(&self, stmts: &mut Stmts) {
        if let Some(god_id) = self.god {
            self.to_stmts_recursive(god_id, stmts)
        }
    }
}
