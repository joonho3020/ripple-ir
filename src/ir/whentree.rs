use chirrtl_parser::ast::*;
use derivative::Derivative;
use petgraph::{graph::{Graph, NodeIndex}, Direction::Outgoing};
use petgraph::Direction::Incoming;
use indexmap::IndexMap;
use indexmap::IndexSet;
use std::{cmp::max, collections::VecDeque, u32, usize};
use std::hash::Hash;

use crate::define_index_type;

define_index_type!(BlockPrior);
define_index_type!(StmtPrior);

/// Represents the priority of an input edge going into a Phi node
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
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

    pub fn bottom() -> Self {
        Self { block: BlockPrior(u32::MIN), stmt: StmtPrior(u32::MIN) }
    }

    pub fn top() -> Self {
        Self { block: BlockPrior(u32::MAX-1), stmt: StmtPrior(u32::MAX-1) }
    }

    pub fn god() -> Self {
        Self { block: BlockPrior(u32::MAX), stmt: StmtPrior(u32::MAX) }
    }

    pub fn inc_block(&mut self) -> BlockPrior {
        self.block.0 += 1;
        self.block
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

/// Higher prior               Lower priority
/// Lower value  <-----------> Higher value
/// Bottom                     Top
impl PartialOrd for PrioritizedCond {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (&self.cond, &other.cond) {
            (Condition::When(se), Condition::Else(oe)) => {
                if se == oe {
                    if self.prior.stmt == other.prior.stmt {
                        // Matching when-else block
                        Some(std::cmp::Ordering::Greater)
                    } else {
                        // Compare between block won't work here as the priority
                        // value of the else is higher when:
                        // 1) the else is on top of the current when
                        // 2) the else is the corresponding case for the current when
                        Some(self.prior.stmt.cmp(&other.prior.stmt))
                    }
                } else {
                    // Different case, fall back to default comparison
                    Some(self.prior.cmp(&other.prior))
                }
            }
            (Condition::Else(se), Condition::When(oe)) => {
                if se == oe {
                    if self.prior.stmt == other.prior.stmt {
                        // Matching when-else block
                        Some(std::cmp::Ordering::Less)
                    } else {
                        // Similar to the above, just compare stmt
                        Some(self.prior.stmt.cmp(&other.prior.stmt))
                    }
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
pub struct PrioritizedCondPath(pub Vec<PrioritizedCond>);

impl PrioritizedCondPath {
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

    pub fn top() -> Self {
        Self(vec![
            PrioritizedCond::new(PhiPrior::top(), Condition::Root)
        ])
    }

    pub fn bottom() -> Self {
        Self(vec![
            PrioritizedCond::new(PhiPrior::bottom(), Condition::Root)
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

    pub fn last(&self) -> Option<&PrioritizedCond> {
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

    pub fn contains(&self, other: &Self) -> bool {
        if self.len() > other.len() {
            return false;
        }
        let other_len = other.len();
        for (i, other_pcond) in other.iter().enumerate() {
            if i == other_len - 1 {
                assert!(other_pcond.cond == Condition::Root);
            } else {
                if self.0[i] != *other_pcond {
                    return false;
                }
            }
        }
        true
    }
}

impl PartialOrd for PrioritizedCondPath {
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

impl Ord for PrioritizedCondPath {
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrioritizedStmt {
    pub stmt: Stmt,
    pub prior: StmtPrior,
}

impl PrioritizedStmt {
    pub fn new(stmt: Stmt, prior: StmtPrior) -> Self {
        Self { stmt, prior }
    }
}

#[derive(Derivative, Default, Clone, Eq)]
#[derivative(Debug)]
pub struct WhenTreeNode {
    /// Condition to reach this node
    pub cond: Condition,

    /// Priority of this node
    /// - prior.block: Block level priority
    /// - prior.stmt: Stmt level priority, useful when there are multiple
    /// whens with the same condition
    pub prior: PhiPrior,

    /// Statements in this tree node
    #[derivative(Debug="ignore")]
    pub stmts: Vec<PrioritizedStmt>
}

impl WhenTreeNode {
    pub fn new(cond: Condition, priority: PhiPrior) -> Self {
        Self { cond, prior: priority, stmts: vec![] }
    }

    pub fn god() -> Self {
        Self::new(Condition::Root, PhiPrior::god())
    }

    pub fn top() -> Self {
        Self::new(Condition::Root, PhiPrior::top())
    }

    pub fn bottom() -> Self {
        Self::new(Condition::Root, PhiPrior::bottom())
    }
}

impl Hash for WhenTreeNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.cond.hash(state);
        self.prior.hash(state);
    }
}

impl PartialEq for WhenTreeNode {
    fn eq(&self, other: &Self) -> bool {
        self.cond == other.cond && self.prior == other.prior
    }
}

/// Map from WhenTreeLeafNodes to their condition path
pub type LeafToPath<'a> = IndexMap<&'a WhenTreeNode, PrioritizedCondPath>;

pub type WhenTreeEdge = ();

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
/// God
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
#[derive(Debug, Clone, Default)]
pub struct WhenTree {
    pub graph: WhenTreeGraph,
    pub god: Option<NodeIndex>
}

impl WhenTree {

    fn corresponding_node(&self, conds: &PrioritizedCondPath) -> Option<NodeIndex> {
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
                    pcond.prior.block == child.prior
                {
                    return Some(cid);
                } else if child.cond == pcond.cond &&
                    child.prior == pcond.prior.block {
                    // Found a matching condition, go down one level in the tree
                    q.push_back(cid);
                    i += 1;
                }
            }
        }
        None
    }

    /// Find all ancestors of a node (including itself), walking up toward the root
    fn collect_ancestors(
        graph: &Graph<WhenTreeNode, WhenTreeEdge>,
        start: NodeIndex,
    ) -> Vec<NodeIndex> {
        let mut ancestors = vec![start];
        let mut current = start;
        while let Some(parent) = graph.neighbors_directed(current, Incoming).next() {
            ancestors.push(parent);
            current = parent;
        }
        ancestors
    }

    fn lowest_common_ancester(&self, conds: &Vec<PrioritizedCondPath>) -> Option<NodeIndex> {
        if conds.is_empty() {
            return None;
        }

        let nodes: Vec<NodeIndex> = conds.iter().map(|c| self.corresponding_node(c).unwrap()).collect();

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

    /// Returns the prioritized cond of this node
    fn prioritized_conds(&self, id: NodeIndex) -> PrioritizedCondPath {
        let mut ancestors = Self::collect_ancestors(&self.graph, id);

        // remove god node
        ancestors.pop();

        // reverse order
        ancestors.reverse();

        // collect PrioritizedConds
        let mut ret = PrioritizedCondPath::default();
        for id in ancestors {
            let node = self.graph.node_weight(id).unwrap();
            let phi_prior = PhiPrior::new(node.prior, StmtPrior(0));
            let pcond = PrioritizedCond::new(phi_prior, node.cond.clone());
            ret.push(pcond);
        }
        return ret;
    }

    /// Find bottom up priority placement
    pub fn bottom_up_priority_constraint(&self, conds: &Vec<PrioritizedCondPath>) -> Option<PrioritizedCondPath> {
        let lca_opt = self.lowest_common_ancester(conds);
        match lca_opt {
            Some(lca) => {
                let node = self.graph.node_weight(lca).unwrap();
                if node.cond == Condition::Root && node.prior != BlockPrior::god() {
                    Some(self.prioritized_conds(lca))
                } else {
                    let childs = self.graph.neighbors_directed(lca, Outgoing);
                    let mut cur_prior = BlockPrior::bottom();
                    let mut id: Option<NodeIndex> = None;
                    for cid in childs {
                        let child = self.graph.node_weight(cid).unwrap();
                        if child.cond == Condition::Root && cur_prior < child.prior {
                            cur_prior = child.prior;
                            id = Some(cid);
                        }
                    }

                    if id.is_some() {
                        Some(self.prioritized_conds(id.unwrap()))
                    } else {
                        None
                    }
                }
            }
            None => {
                None
            }
        }
    }

    /// Follow a given condition (and the priority when it is given) to the tree leaf node
    /// and return a mutable reference to it
    pub fn get_node_mut(
        &mut self,
        conds: &PrioritizedCondPath,
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
                    pcond.prior.block == child.prior
                {
                    if prior.is_some() {
                        // Priority given, check if it matches
                        if child.prior == prior.unwrap().block {
                            return self.graph.node_weight_mut(cid);
                        }
                    } else {
                        // No priority given, just return the first matching one
                        return self.graph.node_weight_mut(cid);
                    }
                } else if child.cond == pcond.cond &&
                    child.prior == pcond.prior.block {
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
        let mut raw_stmt_nodes: Vec<(PrioritizedCondPath, NodeIndex)> = vec![];

        let childs: Vec<NodeIndex> = self.graph.neighbors_directed(id, Outgoing)
                                                .into_iter()
                                                .collect();

        // Collect children grouped by condition expression
        for &cid in childs.iter().rev() {
            let child = self.graph.node_weight(cid).unwrap();
            match &child.cond {
                Condition::Root => {
                    let prior = PhiPrior::new(child.prior, StmtPrior(0));
                    let pconds = PrioritizedCondPath::from_vec(vec![
                        PrioritizedCond::new(prior, Condition::Root)
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

        let mut when_stmts_by_priority: Vec<(PrioritizedCondPath, Stmt)> = vec![];

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
                let prior = PhiPrior::new(when_prior, StmtPrior(0));
                let pconds = PrioritizedCondPath::from_vec(vec![
                    PrioritizedCond::new(prior, Condition::When(expr.clone()))
                ]);
                when_stmts_by_priority.push((pconds, stmt));
            }
        }

        enum RawOrWhen {
            When(Stmt),
            Raw(NodeIndex)
        }

        // Merge raw stmts and when stmts and sort them by priority
        let mut merged: Vec<(PrioritizedCondPath, RawOrWhen)> = vec![];
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

    /// Checks if all conditions in the tree are covered given a vector of PrioritizedConds.
    pub fn is_fully_covered(&self, conds: &Vec<PrioritizedCondPath>) -> bool {
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
