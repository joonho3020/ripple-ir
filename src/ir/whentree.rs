use chirrtl_parser::ast::*;
use derivative::Derivative;
use petgraph::{graph::{Graph, NodeIndex}, Direction::Outgoing};
use indexmap::IndexMap;
use indexmap::IndexSet;
use std::{collections::VecDeque, u32, usize};
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

    pub fn min() -> Self {
        Self::from(u32::MIN)
    }

    pub fn max() -> Self {
        Self::from(u32::MAX)
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

#[derive(Debug, Default, Clone, Eq, Hash)]
pub struct PrioritizedCond {
    /// Priority
    pub prior: PhiPrior,

    /// Condition
    pub cond: Condition,
}

impl PrioritizedCond {
    pub fn new(prior: PhiPrior, cond: Condition) -> Self {
        Self { prior, cond }
    }
}

impl PartialEq for PrioritizedCond {
    fn eq(&self, other: &Self) -> bool {
        self.cond == other.cond && self.prior == other.prior
    }
}

impl PartialOrd for PrioritizedCond {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (&self.cond, &other.cond) {
            (Condition::When(se), Condition::Else(oe)) => {
                if se == oe {
                    Some(std::cmp::Ordering::Greater)
                } else {
                    Some(self.prior.cmp(&other.prior))
                }
            }
            (Condition::Else(se), Condition::When(oe)) => {
                if se == oe {
                    Some(std::cmp::Ordering::Less)
                } else {
                    Some(self.prior.cmp(&other.prior))
                }
            }
            _ => {
                Some(self.prior.cmp(&other.prior))
            }
        }
    }
}

impl Ord for PrioritizedCond {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct PrioritizedConds(Vec<PrioritizedCond>);

impl PrioritizedConds {
    pub fn always_true(&self) -> bool {
        for cond in self.iter().map(|x| &x.cond) {
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

    pub fn get(&self, i: usize) -> &PrioritizedCond {
        self.0.get(i).unwrap()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn root() -> Self {
        Self(vec![
            PrioritizedCond::new(PhiPrior::new(BlockPrior::max(), StmtPrior(0)), Condition::Root)
        ])
    }

    pub fn iter(&self) -> std::slice::Iter<'_, PrioritizedCond> {
        self.0.iter()
    }

    pub fn from_vec(val: Vec<PrioritizedCond>) -> Self {
        Self { 0: val }
    }

    pub fn push(&mut self, pcond: PrioritizedCond) {
        self.0.push(pcond)
    }

    pub fn last_mut(&mut self) -> Option<&mut PrioritizedCond> {
        self.0.last_mut()
    }

    pub fn leaf(&self) -> Option<&PrioritizedCond> {
        self.0.last()
    }

    /// Collect all the selection `Expr` in this condition chain
    pub fn collect_sels(&self) -> Vec<Expr> {
        let mut ret = vec![];
        for cond in self.iter().map(|x| &x.cond) {
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
}

impl PartialOrd for PrioritizedConds {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let len = self.0.len().min(other.0.len());
        for i in 0..len {
            let self_pcond = &self.0[i];
            let other_pcond = &other.0[i];
            if self.0[i] == other.0[i] {
                continue;
            } else {
                return Some(self_pcond.cmp(&other_pcond));
            }
        }

        if self.0.len() > other.0.len() {
            Some(std::cmp::Ordering::Greater)
        } else {
            Some(std::cmp::Ordering::Less)
        }
    }
}

impl Ord for PrioritizedConds {
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

#[derive(Derivative, Default, Clone, Eq)]
#[derivative(Debug)]
pub struct WhenTreeNode {
    /// Condition to reach this node
    pub cond: Condition,

    /// Priority of this node. Smaller means higher priority
    pub priority: BlockPrior,

    /// Statements in this tree node
    #[derivative(Debug="ignore")]
    pub stmts: Stmts
}

impl WhenTreeNode {
    pub fn new(cond: Condition, priority: BlockPrior) -> Self {
        Self { cond, priority, stmts: vec![] }
    }

    pub fn god() -> Self {
        Self::new(Condition::Root, BlockPrior::max())
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
pub type LeafToConditions<'a> = IndexMap<&'a WhenTreeNode, PrioritizedConds>;

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
    pub god: Option<NodeIndex>
}

impl WhenTree {
    pub fn new() -> Self {
        Self { graph: WhenTreeGraph::new(), god: None }
    }

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
        let god_node = WhenTreeNode::god();
        let god_id = self.graph.add_node(god_node);
        self.god = Some(god_id);
        self.from_stmts_recursive(&mut BlockPrior::min(), god_id, stmts);
    }

    /// Returns all the leaf nodes along with the condition path to reach it
    pub fn leaf_to_conditions(&self) -> LeafToConditions {
        let mut ret = LeafToConditions::new();
        let mut q: VecDeque<(NodeIndex, PrioritizedConds)> = VecDeque::new();

        let mut unique_priorities: IndexSet<BlockPrior> = IndexSet::new();

        q.push_back((self.god.unwrap(), PrioritizedConds::default()));

        while !q.is_empty() {
            let nic = q.pop_front().unwrap();

            let mut num_childs = 0;
            for cid in self.graph.neighbors_directed(nic.0, Outgoing) {
                let cnode = self.graph.node_weight(cid).unwrap();
                num_childs += 1;

                let mut path = nic.1.clone();
                let phi_prior = PhiPrior::new(cnode.priority, StmtPrior(0));
                let pcond = PrioritizedCond::new(phi_prior, cnode.cond.clone());
                path.push(pcond);
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
    pub fn from_conditions(cond_paths: Vec<&PrioritizedConds>) -> Self {
        let mut tree = WhenTree::new();

        // Set up the god node
        let god_node = WhenTreeNode::god();
        let god_id = tree.graph.add_node(god_node);
        tree.god = Some(god_id);

        let mut cond_to_node: IndexMap<(BlockPrior, &Condition), NodeIndex> = IndexMap::new();

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
            let mut cur_id = god_id;
            // Traverse down the tree attaching nodes as needed
            for pcond in pconds.iter() {
                if pcond.cond == Condition::Root {
                    if root_cond_nodes.contains(&pcond.prior.block) {
                        continue;
                    } else {
                        // Found a Root condition: add as leaf node if not visited
                        let node = WhenTreeNode::new(pcond.cond.clone(), pcond.prior.block);
                        add_node_and_connect(&mut tree, node, cur_id);
                        root_cond_nodes.insert(pcond.prior.block);
                    }
                } else if !cond_to_node.contains_key(&(pcond.prior.block, &pcond.cond)) {
                    // Condition not yet visited: add to tree
                    let node = WhenTreeNode::new(pcond.cond.clone(), pcond.prior.block);
                    let id = add_node_and_connect(&mut tree, node, cur_id);
                    cond_to_node.insert((pcond.prior.block, &pcond.cond), id);
                    cur_id = id;
                } else {
                    // Already added this condition before
                    cur_id = *cond_to_node.get(&(pcond.prior.block, &pcond.cond)).unwrap();

                }
            }
        }

        // Add a node that will be the default node where stmts under to condition
        // will be inserted into if it already doesn't exist
// let mut has_default_node = false;
// for cid in tree.graph.neighbors_directed(god_id, Outgoing) {
// let child = tree.graph.node_weight(cid).unwrap();
// if child.priority == BlockPrior::min() && child.cond == Condition::Root {
// has_default_node = true;
// break;
// }
// }

// if !has_default_node {
// let id = tree.graph.add_node(WhenTreeNode::new(Condition::Root, BlockPrior::min()));
// tree.graph.add_edge(god_id, id, WhenTreeEdge::default());
// }

        let id = tree.graph.add_node(WhenTreeNode::new(Condition::Root, BlockPrior::max()));
        tree.graph.add_edge(god_id, id, WhenTreeEdge::default());

        tree
    }

    /// Follow a given condition (and the priority when it is given) to the tree leaf node
    /// and return a mutable reference to it
    pub fn get_node_mut(
        &mut self,
        conds: &PrioritizedConds,
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
                    pcond.prior.block == child.priority &&
                    child.cond == Condition::Root
                {
                    if prior.is_some() {
                        if child.priority == prior.unwrap().block {
                            // No priority given, just return the first matching one
                            return self.graph.node_weight_mut(cid);
                        }
                    } else {
                        // No priority given, just return the first matching one
                        return self.graph.node_weight_mut(cid);
                    }
                } else if child.cond == pcond.cond &&
                    child.priority == pcond.prior.block {
                    // Found a matching condition, go down one level in the tree
                    q.push_back(cid);
                    i += 1;
                }
            }
        }
        None
    }

    fn to_stmts_recursive(&self, id: NodeIndex, stmts: &mut Stmts) {
        let mut cond_groups: IndexMap<Expr, (Option<NodeIndex>, Option<NodeIndex>)> = IndexMap::new();
        let mut raw_stmt_nodes: Vec<(PrioritizedConds, NodeIndex)> = vec![];

        let childs: Vec<NodeIndex> = self.graph.neighbors_directed(id, Outgoing)
                                                .into_iter()
                                                .collect();

        // Collect children grouped by condition expression
        for &cid in childs.iter().rev() {
            let child = self.graph.node_weight(cid).unwrap();
            match &child.cond {
                Condition::Root => {
                    let prior = PhiPrior::new(child.priority, StmtPrior(0));
                    let pconds = PrioritizedConds::from_vec(vec![
                        PrioritizedCond::new(prior, Condition::Root)
                    ]);
                    raw_stmt_nodes.push((pconds, cid));
                }
                Condition::When(expr) => {
                    cond_groups.entry(expr.clone()).or_default().0 = Some(cid);
                }
                Condition::Else(expr) => {
                    cond_groups.entry(expr.clone()).or_default().1 = Some(cid);
                }
            }
        }

        let mut when_stmts_by_priority: Vec<(PrioritizedConds, Stmt)> = vec![];

        // Recurse on condition branches and collect them by priority
        for (expr, (when_id_opt, else_id_opt)) in cond_groups {
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
                self.graph.node_weight(when_id).unwrap().priority
            } else {
                // No when stmt: stmts inside when is only a skip
                self.graph.node_weight(else_id_opt.unwrap()).unwrap().priority
            };
            let prior = PhiPrior::new(when_prior, StmtPrior(0));
            let pconds = PrioritizedConds::from_vec(vec![
                PrioritizedCond::new(prior, Condition::When(expr))
            ]);
            when_stmts_by_priority.push((pconds, stmt));
        }

        enum RawOrWhen {
            When(Stmt),
            Raw(NodeIndex)
        }

        // Merge raw stmts and when stmts and sort them by priority
        let mut merged: Vec<(PrioritizedConds, RawOrWhen)> = vec![];
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
        if let Some(god_id) = self.god {
            self.to_stmts_recursive(god_id, stmts)
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
