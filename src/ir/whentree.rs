use chirrtl_parser::ast::*;
use petgraph::{graph::{Graph, NodeIndex}, Direction::Outgoing};
use indexmap::IndexMap;
use crate::ir::PhiPriority;

/// Represents a chain of conditions in a decision tree (a.k.a mux tree)
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Condition {
    /// Root condition (basically always executed)
    #[default]
    Root,

    /// Condition is true when Box<Condition> && Expr
    When(Box<Condition>, Expr),

    /// Condition is true when Box<Condition> && !Expr
    Else(Box<Condition>, Expr),
}

impl Condition {
    /// Collect all the selection `Expr` in this condition chain
    pub fn collect_sels(&self) -> Vec<Expr> {
        let mut ret: Vec<Expr> = vec![];
        match self {
            Self::When(par, expr) |
                Self::Else(par, expr) => {
                ret.append(&mut par.collect_sels());
                ret.push(expr.clone());
            }
            _ => { }
        }
        return ret;
    }

    pub fn always_true(&self) -> bool {
        match self {
            Self::When(_par, _expr) |
            Self::Else(_par, _expr) => {
                false
            }
            _ => {
                true
            }
        }
    }
}


#[derive(Debug, Clone)]
pub struct PrioritizedCond {
    pub prior: PhiPriority,
    pub cond: Condition,
}

impl PrioritizedCond {
    pub fn new(prior: PhiPriority, cond: Condition) -> Self {
        Self { prior, cond }
    }
}

impl Eq for PrioritizedCond {}

impl PartialEq for PrioritizedCond {
    fn eq(&self, other: &Self) -> bool {
        self.cond ==  other.cond && self.prior == other.prior
    }
}

impl Ord for PrioritizedCond {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.cond.collect_sels() == other.cond.collect_sels() {
            match (&self.cond, &other.cond) {
                (Condition::When(..), Condition::Else(..)) => std::cmp::Ordering::Less,
                _ => std::cmp::Ordering::Greater,
            }
        } else {
            self.prior.cmp(&other.prior)
        }
    }
}

impl PartialOrd for PrioritizedCond {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}


#[derive(Default, Debug, Clone, PartialEq)]
pub struct WhenTreeNode {
    /// Condition to reach this node
    pub cond: Condition,

    /// Priority of this node. Smaller means higher priority
    pub priority: u32,

    /// Statements in this tree node
    pub stmts: Stmts
}

impl WhenTreeNode {
    pub fn new(cond: Condition, priority: u32) -> Self {
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

        for child in self.graph.neighbors_directed(id, Outgoing) {
            self.print_tree_recursive(child, depth + 1);
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
        parent_cond: Condition,
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

                    self.from_stmts_recursive(
                        parent_priority,
                        Condition::When(Box::new(parent_cond.clone()), cond.clone()),
                        parent_id,
                        when_stmts);

                    *parent_priority += 1;

                    if let Some(else_stmts) = else_stmts_opt {
                        self.from_stmts_recursive(
                            parent_priority,
                            Condition::Else(Box::new(parent_cond.clone()), cond.clone()),
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
        let root_node = WhenTreeNode::new(Condition::Root, 0);
        let root_id = self.graph.add_node(root_node);
        self.root = Some(root_id);
        self.from_stmts_recursive(&mut 0, Condition::Root, root_id, stmts);
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
        let root_node = WhenTreeNode::new(Condition::Root, 0);
        let root_id = tree.graph.add_node(root_node);
        tree.root = Some(root_id);

        let mut cond_to_node: IndexMap<Condition, NodeIndex> = IndexMap::new();
        cond_to_node.insert(Condition::Root, root_id);

        // Sort conditions by PhiPriority
        let mut sorted_paths = cond_paths;
        sorted_paths.sort();

        println!("sorted_paths {:?}", sorted_paths);

        for prior_cond in sorted_paths {
            let mut cur_cond = Condition::Root.clone();
            let mut cur_node = root_id;

            let cond = prior_cond.cond;
            let prior = prior_cond.prior;

            // Traverse the chain of conditions
            for expr in cond.collect_sels() {
                let next_cond = match &cond {
                    Condition::When(_, _) => Condition::When(Box::new(cur_cond.clone()), expr.clone()),
                    Condition::Else(_, _) => Condition::Else(Box::new(cur_cond.clone()), expr.clone()),
                    _ => unreachable!(),
                };

                let next_node = *cond_to_node.entry(next_cond.clone()).or_insert_with(|| {
                    let new_node = WhenTreeNode::new(next_cond.clone(), prior.block);
                    let new_node_id = tree.graph.add_node(new_node);
                    tree.graph.add_edge(cur_node, new_node_id, ());
                    new_node_id
                });

                cur_node = next_node;
                cur_cond = next_cond;
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
