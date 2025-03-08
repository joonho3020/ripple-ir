use crate::parser::ast::*;
use indextree::{Arena, NodeId};


#[derive(Default, Debug, Clone, PartialEq, Hash)]
pub enum Condition {
    #[default]
    True,
    When(Box<Condition>, Expr),
    Else(Box<Condition>, Expr),
}

impl Condition {
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
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct WhenTreeNode {
    pub cond: Condition,
    pub priority: u32,
    pub stmts: Stmts
}

impl WhenTreeNode {
    pub fn new(cond: Condition, priority: u32) -> Self {
        Self { cond, priority, stmts: vec![] }
    }
}

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

    pub fn from_stmts(&mut self, stmts: &Stmts) {
        let root_node = WhenTreeNode::new(Condition::True, 0);
        let root_id = self.arena.new_node(root_node);
        self.root = Some(root_id);
        self.from_stmts_recursive(&mut 0, Condition::True, root_id, stmts);
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
    use crate::parser::parse_circuit;

    #[test]
    fn gcd() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./examples/GCD.fir")?;
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
    fn nested_when() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./examples/NestedWhen.fir")?;
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
}
