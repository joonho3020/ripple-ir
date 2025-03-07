use crate::parser::{ast::*, Int};
use indextree::{Arena, NodeId};
use std::{collections::VecDeque, fmt::Debug};

#[derive(Default, Debug, Clone, Copy, PartialEq, Hash)]
pub enum Direction {
    #[default]
    Output,
    Input,
}

impl Direction {
    pub fn flip(&self) -> Self {
        match self {
            Self::Input => Self::Output,
            Self::Output => Self::Input,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct TypeTreeNode {
    pub name: Identifier,
    pub dir: Direction,
    pub tpe: Option<TypeGround>,
}

impl TypeTreeNode {
    pub fn new(name: Identifier, dir: Direction, tpe: Option<TypeGround>) -> Self {
        Self { name, dir, tpe }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeTree {
    pub arena: Arena<TypeTreeNode>,
    pub root: Option<NodeId>
}

impl TypeTree {
    pub fn new() -> Self {
        Self { arena: Arena::new(), root: None }
    }

    pub fn get(&self, id: NodeId) -> &TypeTreeNode {
        self.arena[id].get()
    }

    pub fn get_mut(&mut self, id: NodeId) -> &mut TypeTreeNode {
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

    fn collect_leaf_nodes_recursive(&self, node: NodeId, leaf_nodes: &mut Vec<NodeId>) {
        if node.children(&self.arena).next().is_none() {
            leaf_nodes.push(node);
        } else {
            for child in node.children(&self.arena) {
                self.collect_leaf_nodes_recursive(child, leaf_nodes);
            }
        }
    }

    pub fn collect_leaf_nodes(&self) -> Vec<NodeId> {
        let mut leaf_nodes = Vec::new();
        match self.root {
            Some(rid) => {
                self.collect_leaf_nodes_recursive(rid, &mut leaf_nodes);
            }
            _ => { }
        }
        leaf_nodes
    }

    pub fn all_possible_references(&self) -> Vec<Reference> {
        let mut ret = vec![];
        let mut q: VecDeque<(NodeId, Reference)> = VecDeque::new();
        if let Some(root_id) = self.root {
            let root = self.get(root_id);
            q.push_back((root_id, Reference::Ref(root.name.clone())));
        }
        while !q.is_empty() {
            let (id, cur_ref) = q.pop_front().unwrap();
            ret.push(cur_ref.clone());

            for child_id in id.children(&self.arena) {
                let child = self.get(child_id);
                let child_ref = match &child.name {
                    Identifier::ID(id) => {
                        Reference::RefIdxInt(Box::new(cur_ref.clone()), id.clone())
                    }
                    Identifier::Name(_) => {
                        Reference::RefDot(Box::new(cur_ref.clone()), child.name.clone())
                    }
                };
                q.push_back((child_id, child_ref));
            }
        }
        return ret;
    }
}

impl Type {
    fn create_node_and_add_to_parent(&self,
        type_ground: Option<TypeGround>,
        name: Identifier,
        dir: Direction,
        parent_opt: Option<NodeId>,
        tree: &mut TypeTree
    ) -> NodeId {
        let child = tree.arena.new_node(TypeTreeNode::new(name, dir, type_ground));
        match parent_opt {
            Some(parent) => {
                parent.append(child, &mut tree.arena);
            }
            None => {
                tree.root = Some(child);
                // No parent, must have been root node
            }
        }
        child
    }

    fn construct_tree_recursive(&self,
        name: Identifier,
        dir: Direction,
        parent_opt: Option<NodeId>,
        tree: &mut TypeTree
    ) {
        match self {
            Type::TypeGround(tg) => {
                self.create_node_and_add_to_parent(
                    Some(tg.clone()), name, dir, parent_opt, tree);
            }
            Type::TypeAggregate(ta) => {
                let node_id = self.create_node_and_add_to_parent(
                    None, name, dir, parent_opt, tree);

                match ta.as_ref() {
                    TypeAggregate::Fields(fields) => {
                        for field in fields.iter() {
                            match field.as_ref() {
                                Field::Straight(name, tpe) => {
                                    tpe.construct_tree_recursive(
                                        name.clone(), dir, Some(node_id), tree);
                                }
                                Field::Flipped(name, tpe) => {
                                    tpe.construct_tree_recursive(
                                        name.clone(), dir.flip(), Some(node_id), tree);
                                }
                            }
                        }
                    }
                    TypeAggregate::Array(tpe, len) => {
                        for id in 0..len.to_u32() {
                            tpe.construct_tree_recursive(
                                Identifier::ID(Int::from(id)), dir, Some(node_id), tree);
                        }
                    }
                }
            }
            _ => {
                panic!("Unrecognized type while constructing type tree {:?}", self);
            }
        }
    }

    pub fn construct_tree(&self, name: Identifier, dir: Direction) -> TypeTree {
        let mut ret = TypeTree::new();
        self.construct_tree_recursive(name, dir, None, &mut ret);
        return ret;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::parser::parse_circuit;

    #[test]
    fn print_type_tree() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./examples/NestedBundleModule.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for port in m.ports.iter() {
                        let port_type_tree = match port.as_ref() {
                            Port::Input(name, tpe, info) => {
                                tpe.construct_tree(name.clone(), Direction::Input)
                            }
                            Port::Output(name, tpe, info) => {
                                tpe.construct_tree(name.clone(), Direction::Output)
                            }
                        };
                        port_type_tree.print_tree();
                    }

                }
                CircuitModule::ExtModule(e) => {
                }
            }
        }
        return Ok(());
    }
}
