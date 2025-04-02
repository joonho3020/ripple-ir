use chirrtl_parser::ast::*;
use petgraph::{graph::{Graph, NodeIndex}, Direction::Outgoing};
use indexmap::IndexMap;

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct PhiPriority {
    /// Priority between blocks
    /// - Smaller number means higher priority
    pub block: u32,

    /// Priority between statements within the same block
    /// - Smaller number means higher priority
    pub stmt: u32,
}

impl PhiPriority {
    pub fn new(block: u32, stmt: u32) -> Self {
        Self { block, stmt }
    }
}

impl Ord for PhiPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.block == other.block {
            self.stmt.cmp(&other.stmt)
        } else {
            self.block.cmp(&other.block)
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrioritizedCond {
    pub prior: PhiPriority,
    pub conds: Conditions,
}

impl PrioritizedCond {
    pub fn new(prior: PhiPriority, conds: Conditions) -> Self {
        Self { prior, conds: conds }
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
            match self.conds.path.get(idx).unwrap() {
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
pub struct Conditions {
    path: Vec<Condition>
}

impl Conditions {
    /// Collect all the selection `Expr` in this condition chain
    pub fn collect_sels(&self) -> Vec<Expr> {
        let mut ret = vec![];
        for cond in self.path.iter() {
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

    pub fn always_true(&self) -> bool {
        for cond in self.path.iter() {
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
        Self { path }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhenTreeNode {
    /// Condition to reach this node
    pub cond: Conditions,

    /// Priority of this node. Smaller means higher priority
    pub priority: u32,

    /// Statements in this tree node
    pub stmts: Stmts
}

impl WhenTreeNode {
    pub fn new(cond: Conditions, priority: u32) -> Self {
        Self { cond, priority, stmts: vec![] }
    }
}

pub type WhenTreeGraph = Graph<WhenTreeNode, ()>;

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

    pub fn get(&self, id: NodeIndex) -> &WhenTreeNode {
        self.graph.node_weight(id).unwrap()
    }

    pub fn get_mut(&mut self, id: NodeIndex) -> &mut WhenTreeNode {
        self.graph.node_weight_mut(id).unwrap()
    }

    fn print_tree_recursive(&self, id: NodeIndex, depth: usize) {
        println!("{}{:?}", "  ".repeat(depth), self.get(id));

        let childs: Vec<NodeIndex> = self.graph.neighbors_directed(id, Outgoing).into_iter().collect();
        for child in childs.iter().rev() {
            self.print_tree_recursive(*child, depth + 1);
        }
    }

    pub fn print_tree(&self) {
        match self.root {
            Some(rid) => {
                self.print_tree_recursive(rid, 0);
            }
            _ => {
                println!("TypeTree empty");
            }
        }
    }

    fn from_stmts_recursive(
        &mut self,
        parent_priority: &mut u32,
        parent_cond: Conditions,
        parent_id: NodeIndex,
        stmts: &Stmts,
    ) {
        let mut cur_node: Option<&mut WhenTreeNode> = None;

        // Traverse in reverse order to take last-connection semantics into account
        for stmt in stmts.iter().rev() {
            match stmt.as_ref() {
                Stmt::When(cond, _info, when_stmts, else_stmts_opt) => {
                    *parent_priority += 1;
                    cur_node = None;

                    let when_cond = Condition::When(cond.clone());
                    let mut conds = parent_cond.clone();
                    conds.path.push(when_cond);

                    self.from_stmts_recursive(
                        parent_priority,
                        conds,
                        parent_id,
                        when_stmts);

                    *parent_priority += 1;

                    if let Some(else_stmts) = else_stmts_opt {
                        let else_cond = Condition::Else(cond.clone());
                        let mut conds = parent_cond.clone();
                        conds.path.push(else_cond);

                        self.from_stmts_recursive(
                            parent_priority,
                            conds,
                            parent_id,
                            else_stmts);

                        *parent_priority += 1;
                    }
                }
                _ => {
                    match cur_node.as_mut() {
                        None => {
                            // Add node to parent
                            let tn = WhenTreeNode::new(parent_cond.clone(), *parent_priority);
                            let child_id = self.graph.add_node(tn);
                            self.graph.add_edge(parent_id, child_id, ());
                            cur_node = Some(self.get_mut(child_id));
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
        let root_node = WhenTreeNode::new(Conditions::default(), 0);
        let root_id = self.graph.add_node(root_node);
        self.root = Some(root_id);
        self.from_stmts_recursive(&mut 0, Conditions::default(), root_id, stmts);
    }

    fn collect_leaf_nodes_recursive(&self, id: NodeIndex, leaf_ids: &mut Vec<NodeIndex>) {
        if self.graph.neighbors_directed(id, Outgoing).count() == 0 {
            leaf_ids.push(id);
        } else {
            for child in self.graph.neighbors_directed(id, Outgoing) {
                self.collect_leaf_nodes_recursive(child, leaf_ids);
            }
        }
    }

    /// Returns all the leaf nodes
    pub fn leaf_nodes(&self) -> Vec<&WhenTreeNode> {
        let mut leaf_nodes = Vec::new();
        match self.root {
            Some(rid) => {
                self.collect_leaf_nodes_recursive(rid, &mut leaf_nodes);
            }
            _ => { }
        }
        leaf_nodes.iter().map(|id| self.get(*id)).collect()
    }

    /// Reconstructs a WhenTree from a vector of (priority, condition) pairs
    pub fn from_conditions(cond_paths: Vec<PrioritizedCond>) -> Self {
        let mut tree = WhenTree::new();

        // Set up the root node
        let root_node = WhenTreeNode::new(Conditions::default(), 0);
        let root_id = tree.graph.add_node(root_node);
        tree.root = Some(root_id);

        let mut cond_to_node: IndexMap<Condition, NodeIndex> = IndexMap::new();
        let mut sorted_paths = cond_paths;
        sorted_paths.sort_by(|a, b| b.cmp(a));

        println!("sorted_path {:?}", sorted_paths);

        for pconds in sorted_paths {
            let mut cur_id = root_id;
            let mut cur_conds = vec![];
            for cur_cond in pconds.conds.path.iter() {
                cur_conds.push(cur_cond.clone());
                match cur_cond {
                    Condition::Root => {
                        assert!(cur_id == root_id);
                        let id = tree.graph.add_node(
                            WhenTreeNode::new(
                                Conditions::from_vec(cur_conds.clone()),
                                pconds.prior.block));
                        tree.graph.add_edge(cur_id, id, ());
                        cur_id = id;
                    }
                    _ => {
                        cur_id = if !cond_to_node.contains_key(cur_cond) {
                            let id = tree.graph.add_node(
                                WhenTreeNode::new(
                                    Conditions::from_vec(cur_conds.clone()),
                                    pconds.prior.block));
                            tree.graph.add_edge(cur_id, id, ());
                            cond_to_node.insert(cur_cond.clone(), id);
                            id
                        } else {
                            *cond_to_node.get(cur_cond).unwrap()
                        };
                    }
                }
            }
            // We don’t push any actual statements, so nodes are empty.
            // But we now have the correct hierarchy in place.
        }


        tree
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
