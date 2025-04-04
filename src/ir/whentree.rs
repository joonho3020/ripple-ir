use chirrtl_parser::ast::*;
use derivative::Derivative;
use petgraph::{graph::{Graph, NodeIndex}, Direction::Outgoing};
use indexmap::IndexMap;
use indexmap::IndexSet;
use std::{collections::VecDeque, usize};
use std::hash::Hash;

use crate::define_index_type;

define_index_type!(BlockPrior);

impl BlockPrior {
    pub fn increment(&mut self) {
        self.0 += 1
    }

    pub fn one() -> Self {
        Self::from(1u32)
    }
}

define_index_type!(StmtPrior);

/// Represents the priority of an input edge going into a Phi node
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct PhiPrior {
    /// Priority between blocks
    /// - Smaller number means higher priority
    pub block: BlockPrior,

    /// Priority between statements within the same block
    /// - Smaller number means higher priority
    pub stmt: StmtPrior,
}

impl PhiPrior {
    pub fn new(block: BlockPrior, stmt: StmtPrior) -> Self {
        Self { block, stmt }
    }
}

impl Ord for PhiPrior {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.block == other.block {
            self.stmt.cmp(&other.stmt)
        } else {
            self.block.cmp(&other.block)
        }
    }
}

#[derive(Derivative, Default, Clone, Hash)]
#[derivative(Debug)]
pub struct PrioritizedCond {
    /// Priority
    pub prior: PhiPrior,

    /// Condition chain
    #[derivative(Debug="ignore")] 
    pub conds: Conditions,
}

impl PrioritizedCond {
    pub fn new(prior: PhiPrior, conds: Conditions) -> Self {
        Self { prior, conds }
    }
}

impl Eq for PrioritizedCond {}

impl PartialEq for PrioritizedCond {
    fn eq(&self, other: &Self) -> bool {
        self.conds == other.conds && self.prior == other.prior
    }
}

impl PartialOrd for PrioritizedCond {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        fn last_continuous_match<T: PartialEq>(a: &[T], b: &[T]) -> Option<usize> {
            let len = a.len().min(b.len());
            if len == 0 {
                return None;
            }
            for i in 0..len {
                if a[i] != b[i] {
                    return if i == 0 { None } else { Some(i - 1) };
                }
            }
            Some(len - 1)
        }

        let self_sel_path = self.conds.collect_sels();
        let other_sel_path = other.conds.collect_sels();
        if let Some(idx) = last_continuous_match(&self_sel_path, &other_sel_path) {
            match self.conds.path().get(idx).unwrap() {
                Condition::When(..) => Some(std::cmp::Ordering::Greater),
                Condition::Else(..) => Some(std::cmp::Ordering::Less),
                _ => { Some(self.prior.cmp(&other.prior)) }
            }
        } else {
            Some(self.prior.cmp(&other.prior))
        }
    }
}

impl Ord for PrioritizedCond {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// When/Else conditions
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Condition {
    /// Root condition (basically always executed)
    #[default]
    Root,

    /// Condition is true when Box<Condition> && Expr
    When(Expr),

    /// Condition is true when Box<Condition> && !Expr
    Else(Expr),
}

/// Represents a chain of conditions in a decision tree (a.k.a mux tree)
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Conditions(Vec<Condition>);

impl Conditions {
    pub fn root() -> Self {
        Self(vec![Condition::Root])
    }

    pub fn path(&self) -> &Vec<Condition> {
        &self.0
    }

    pub fn push(&mut self, c: Condition) {
        self.0.push(c)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get(&self, i: usize) -> &Condition {
        &self.0[i]
    }
}

impl Conditions {
    /// Collect all the selection `Expr` in this condition chain
    pub fn collect_sels(&self) -> Vec<Expr> {
        let mut ret = vec![];
        for cond in self.path().iter() {
            match cond {
                Condition::When(e) |
                Condition::Else(e) => {
                    ret.push(e.clone())
                }
                _ => { }
            }
        }
        return ret;
    }

    /// Checks whether the condition chain is always true
    pub fn always_true(&self) -> bool {
        for cond in self.path().iter() {
            match cond {
                Condition::When(..) |
                Condition::Else(..) => {
                    return false;
                }
                _ => {
                    continue;
                }
            }
        }
        return true;
    }

    pub fn from_vec(path: Vec<Condition>) -> Self {
        Self { 0: path }
    }
}

#[derive(Debug, Default, Clone, Eq)]
pub struct WhenTreeNode {
    /// Condition to reach this node
    pub cond: Condition,

    /// Priority of this node. Smaller means higher priority
    pub priority: BlockPrior,

    /// Statements in this tree node
    pub stmts: Stmts
}

impl WhenTreeNode {
    pub fn new(cond: Condition, priority: BlockPrior) -> Self {
        Self { cond, priority, stmts: vec![] }
    }
}

impl Hash for WhenTreeNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.cond.hash(state);
        self.priority.hash(state);
    }
}

impl PartialEq for WhenTreeNode {
    fn eq(&self, other: &Self) -> bool {
        self.cond == other.cond && self.priority == other.priority
    }
}

/// Map from WhenTreeLeafNodes to their condition path
pub type LeafToConditions<'a> = IndexMap<&'a WhenTreeNode, Conditions>;

type WhenTreeEdge = ();

pub type WhenTreeGraph = Graph<WhenTreeNode, WhenTreeEdge>;

/// Represents a tree of decision blocks
///
/// ```
/// stmt_5
/// when (a)
///   stmt_0
///   when (b)
///     stmt_1
///   else
///     stmt_2
/// else
///   stmt_3
/// stmt_4
/// ```
///
/// The above when statements will produce a tree like this:
///
/// ```
/// Root
/// |- stmt_4 (highest priority)
/// |- when (a)
/// |  |- when (b)
/// |  |  |- stmt_1
/// |  |- else
/// |  |  |- stmt_2
/// |  |stmt_0
/// |- else
/// |  |- stmt_3
/// |- stmt_5 (lowest priority)
/// ```
#[derive(Debug, Clone)]
pub struct WhenTree {
    pub graph: WhenTreeGraph,
    pub root: Option<NodeIndex>
}

impl WhenTree {
    pub fn new() -> Self {
        Self { graph: WhenTreeGraph::new(), root: None }
    }

    fn print_tree_recursive(&self, id: NodeIndex, depth: usize) {
        println!("{}{:?}", "  ".repeat(depth), self.graph.node_weight(id).unwrap());

        let childs: Vec<NodeIndex> = self.graph.neighbors_directed(id, Outgoing)
                                                .into_iter()
                                                .collect();
        for child in childs.iter().rev() {
            self.print_tree_recursive(*child, depth + 1);
        }
    }

    /// Print the tree to stdout
    pub fn print_tree(&self) {
        match self.root {
            Some(rid) => {
                self.print_tree_recursive(rid, 0);
            }
            _ => {
                println!("WhenTree empty");
            }
        }
    }

    fn from_stmts_recursive(
        &mut self,
        parent_priority: &mut BlockPrior,
        parent_id: NodeIndex,
        stmts: &Stmts,
    ) {
        let mut cur_node: Option<&mut WhenTreeNode> = None;

        // Traverse in reverse order to take last-connection semantics into account
        for stmt in stmts.iter().rev() {
            match stmt.as_ref() {
                Stmt::When(cond, _info, when_stmts, else_stmts_opt) => {
                    parent_priority.increment();
                    cur_node = None;

                    let when_cond = Condition::When(cond.clone());
                    let id = self.graph.add_node(WhenTreeNode::new(when_cond, *parent_priority));
                    self.graph.add_edge(parent_id, id, WhenTreeEdge::default());
                    self.from_stmts_recursive(
                        parent_priority,
                        id,
                        when_stmts);

                    parent_priority.increment();

                    if let Some(else_stmts) = else_stmts_opt {
                        let else_cond = Condition::Else(cond.clone());
                        let id = self.graph.add_node(WhenTreeNode::new(else_cond, *parent_priority));
                        self.graph.add_edge(parent_id, id, WhenTreeEdge::default());
                        self.from_stmts_recursive(
                            parent_priority,
                            id,
                            else_stmts);

                        parent_priority.increment();
                    }
                }
                _ => {
                    match cur_node.as_mut() {
                        None => {
                            // Add node to parent
                            let tn = WhenTreeNode::new(Condition::Root, *parent_priority);
                            let child_id = self.graph.add_node(tn);
                            self.graph.add_edge(parent_id, child_id, WhenTreeEdge::default());
                            cur_node = Some(self.graph.node_weight_mut(child_id).unwrap());
                        }
                        _ => {}
                    }
                    cur_node.as_mut().unwrap().stmts.push(stmt.clone());
                }
            }
        }
    }

    /// Creates a when tree from given `Stmts`
    pub fn from_stmts(&mut self, stmts: &Stmts) {
        let root_node = WhenTreeNode::new(Condition::Root, BlockPrior(0));
        let root_id = self.graph.add_node(root_node);
        self.root = Some(root_id);
        self.from_stmts_recursive(&mut BlockPrior(1), root_id, stmts);
    }

    /// Returns all the leaf nodes along with the condition path to reach it
    pub fn leaf_to_conditions(&self) -> LeafToConditions {
        let mut ret = LeafToConditions::new();
        let mut q: VecDeque<(NodeIndex, Conditions)> = VecDeque::new();

        let mut unique_priorities: IndexSet<BlockPrior> = IndexSet::new();

        q.push_back((self.root.unwrap(), Conditions::default()));

        while !q.is_empty() {
            let nic = q.pop_front().unwrap();

            let mut num_childs = 0;
            for cid in self.graph.neighbors_directed(nic.0, Outgoing) {
                let cnode = self.graph.node_weight(cid).unwrap();
                num_childs += 1;

                let mut path = nic.1.clone();
                path.push(cnode.cond.clone());
                q.push_back((cid, path));
            }
            let node = self.graph.node_weight(nic.0).unwrap();
            if num_childs == 0 {
                // Leaf node
                ret.insert(node, nic.1);

                // Check for uniqueness of leaf node's priority
                assert!(!unique_priorities.contains(&node.priority));
                unique_priorities.insert(node.priority);
            } else {
                // Non leaf node, should not contain statement
                assert!(node.stmts.is_empty());
            }
        }
        return ret;
    }

    /// Reconstructs a WhenTree from a vector of (priority, condition) pairs
    pub fn from_conditions(cond_paths: Vec<&PrioritizedCond>) -> Self {
        let mut tree = WhenTree::new();

        // Set up the root node
        let root_node = WhenTreeNode::new(Condition::Root, BlockPrior(0));
        let root_id = tree.graph.add_node(root_node);
        tree.root = Some(root_id);

        let mut cond_to_node: IndexMap<Condition, NodeIndex> = IndexMap::new();

        // Root conditions can be indexed by its unique priority
        let mut root_cond_nodes: IndexSet<BlockPrior> = IndexSet::new();

        // Sort by priority
        let mut sorted_paths = cond_paths;
        sorted_paths.sort_by(|a, b| b.cmp(a));

        fn add_node_and_connect(
            tree: &mut WhenTree,
            node: WhenTreeNode,
            parent_id: NodeIndex
        ) -> NodeIndex {
            let id = tree.graph.add_node(node);
            tree.graph.add_edge(parent_id, id, WhenTreeEdge::default());
            id
        }

        // Iterate over found condition paths
        for pconds in sorted_paths {
            let mut cur_id = root_id;
            // Traverse down the tree attaching nodes as needed
            for cur_cond in pconds.conds.path().iter() {
                if *cur_cond == Condition::Root {
                    if root_cond_nodes.contains(&pconds.prior.block) {
                        continue;
                    } else {
                        // Found a Root condition: add as leaf node if not visited
                        let node = WhenTreeNode::new(cur_cond.clone(), pconds.prior.block);
                        add_node_and_connect(&mut tree, node, cur_id);
                        root_cond_nodes.insert(pconds.prior.block);
                    }
                } else if !cond_to_node.contains_key(cur_cond) {
                    // Condition not yet visited: add to tree
                    let node = WhenTreeNode::new(cur_cond.clone(), pconds.prior.block);
                    let id = add_node_and_connect(&mut tree, node, cur_id);
                    cond_to_node.insert(cur_cond.clone(), id);
                    cur_id = id;
                } else {
                    // Already added this condition before
                    cur_id = *cond_to_node.get(cur_cond).unwrap();
                }
            }
        }

        // Add a node that will be the default node where stmts under to condition
        // will be inserted into if it already doesn't exist
        let mut has_default_node = false;
        for root_cid in tree.graph.neighbors_directed(root_id, Outgoing) {
            let child = tree.graph.node_weight(root_cid).unwrap();
            if child.priority == BlockPrior::one() && child.cond == Condition::Root {
                has_default_node = true;
                break;
            }
        }

        if !has_default_node {
            let id = tree.graph.add_node(WhenTreeNode::new(Condition::Root, BlockPrior::one()));
            tree.graph.add_edge(root_id, id, WhenTreeEdge::default());
        }

        tree
    }

    pub fn add_condition(&mut self, conds: &Conditions) {
        let mut q: VecDeque<NodeIndex> = VecDeque::new();
        q.push_back(self.root.unwrap());

        let mut i = 0;
        while !q.is_empty() && i < conds.len() {
            let id = q.pop_front().unwrap();
            let mut found_match = false;
            for cid in self.graph.neighbors_directed(id, Outgoing) {
                let cur_cond = conds.get(i);
                let child = self.graph.node_weight(cid).unwrap();

                // Found root node
                if cur_cond == &Condition::Root && child.cond == Condition::Root {
                    return;
                } else if &child.cond == cur_cond {
                    // Found a matching condition, go down one level in the tree
                    q.push_back(cid);
                    i += 1;
                    found_match = true;
                }
            }

            if !found_match {
                let cur_cond = conds.get(i);
                assert!(cur_cond == &Condition::Root);
                let parent = self.graph.node_weight(id).unwrap();
                let cid = self.graph.add_node(WhenTreeNode::new(Condition::Root, parent.priority));
                self.graph.add_edge(id, cid, WhenTreeEdge::default());
            }
        }
    }

    /// Follow a given condition (and the priority when it is given) to the tree leaf node
    /// and return a mutable reference to it
    pub fn get_node_mut(
        &mut self,
        conds: &Conditions,
        prior: Option<&PhiPrior>
    ) -> Option<&mut WhenTreeNode> {
        let mut q: VecDeque<NodeIndex> = VecDeque::new();
        q.push_back(self.root.unwrap());

        let mut i = 0;
        while !q.is_empty() && i < conds.len() {
            let id = q.pop_front().unwrap();
            for cid in self.graph.neighbors_directed(id, Outgoing) {
                let cur_cond = conds.get(i);
                let child = self.graph.node_weight(cid).unwrap();

                // Found root node
                if cur_cond == &Condition::Root && child.cond == Condition::Root {
                    if prior.is_some() {
                        if child.priority == prior.unwrap().block {
                            // No priority given, just return the first matching one
                            return self.graph.node_weight_mut(cid);
                        }
                    } else {
                        // No priority given, just return the first matching one
                        return self.graph.node_weight_mut(cid);
                    }
                } else if &child.cond == cur_cond {
                    // Found a matching condition, go down one level in the tree
                    q.push_back(cid);
                    i += 1;
                }
            }
        }
        None
    }

    fn to_stmts_recursive(&self, id: NodeIndex, stmts: &mut Stmts) {
        let mut cond_groups: IndexMap<Expr, (NodeIndex, Option<NodeIndex>)> = IndexMap::new();
        let mut raw_stmt_nodes: Vec<(PrioritizedCond, NodeIndex)> = vec![];

        let childs: Vec<NodeIndex> = self.graph.neighbors_directed(id, Outgoing)
                                                .into_iter()
                                                .collect();

        // Collect children grouped by condition expression
        for &cid in childs.iter().rev() {
            let child = self.graph.node_weight(cid).unwrap();
            match &child.cond {
                Condition::Root => {
                    let prior = PhiPrior::new(child.priority, StmtPrior(0));
                    let pcond = PrioritizedCond::new(prior, Conditions::root());
                    raw_stmt_nodes.push((pcond, cid));
                }
                Condition::When(expr) => {
                    cond_groups.entry(expr.clone()).or_default().0 = cid;
                }
                Condition::Else(expr) => {
                    cond_groups.entry(expr.clone()).or_default().1 = Some(cid);
                }
            }
        }

        let mut when_stmts_by_priority: Vec<(PrioritizedCond, Stmt)> = vec![];

        // Recurse on condition branches and collect them by priority
        for (expr, (when_id, else_id_opt)) in cond_groups {
            let mut when_stmts = Stmts::new();
            self.to_stmts_recursive(when_id, &mut when_stmts);

            let else_stmts_opt =  if let Some(else_id) = else_id_opt {
                let mut else_stmts = Stmts::new();
                self.to_stmts_recursive(else_id, &mut else_stmts);
                Some(else_stmts)
            } else {
                None
            };

            let stmt = Stmt::When(expr.clone(), Info::default(), when_stmts, else_stmts_opt);
            let when_prior = self.graph.node_weight(when_id).unwrap().priority;
            let prior = PhiPrior::new(when_prior, StmtPrior(0));
            let pcond = PrioritizedCond::new(prior, Conditions::from_vec(vec![Condition::When(expr)]));
            when_stmts_by_priority.push((pcond, stmt));
        }

        enum RawOrWhen {
            When(Stmt),
            Raw(NodeIndex)
        }

        // Merge raw stmts and when stmts and sort them by priority
        let mut merged: Vec<(PrioritizedCond, RawOrWhen)> = vec![];
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
                        stmts.push(stmt.clone());
                    }
                }
            }
        }
    }

    /// Reconstructs FIRRTL `Stmts` from the WhenTree.
    pub fn to_stmts(&self, stmts: &mut Stmts) {
        if let Some(root_id) = self.root {
            self.to_stmts_recursive(root_id, stmts)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use chirrtl_parser::parse_circuit;
    use crate::common::RippleIRErr;

    fn run(path: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(path)?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for cm in circuit.modules {
            let mut whentree = WhenTree::new();
            match cm.as_ref() {
                CircuitModule::Module(m) => {
                    whentree.from_stmts(m.stmts.as_ref());
                    whentree.print_tree();
                }
                _ => { }
            }
        }
        Ok(())
    }

    #[test]
    fn gcd() {
        run("./test-inputs/GCD.fir").expect("gcd");
    }

    #[test]
    fn nested_when() {
        run("./test-inputs/NestedWhen.fir").expect("nested_when");
    }
}
