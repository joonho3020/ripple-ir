use chirrtl_parser::ast::*;
use indextree::{Arena, NodeId};

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
#[derive(Debug, Clone, PartialEq)]
pub struct WhenTree {
    pub arena: Arena<WhenTreeNode>,
    pub root: Option<NodeId>
}

impl WhenTree {
    pub fn new() -> Self {
        Self { arena: Arena::new(), root: None }
    }

    pub fn get(&self, id: NodeId) -> &WhenTreeNode {
        self.arena[id].get()
    }

    pub fn get_mut(&mut self, id: NodeId) -> &mut WhenTreeNode {
        self.arena[id].get_mut()
    }

    fn print_tree_recursive(&self, node: NodeId, depth: usize) {
        println!("{}{:?}", "  ".repeat(depth), self.arena[node].get());

        for child in node.children(&self.arena) {
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
        parent_id: NodeId,
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
                            let child = self.arena.new_node(tn);
                            parent_id.append(child, &mut self.arena);

                            cur_node = Some(self.get_mut(child));
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
        let root_id = self.arena.new_node(root_node);
        self.root = Some(root_id);
        self.from_stmts_recursive(&mut 0, Condition::Root, root_id, stmts);
    }

    fn collect_leaf_nodes_recursive(&self, node: NodeId, leaf_nodes: &mut Vec<NodeId>) {
        if node.children(&self.arena).next().is_none() {
            leaf_nodes.push(node);
        } else {
            for child in node.children(&self.arena) {
                self.collect_leaf_nodes_recursive(child, leaf_nodes);
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
