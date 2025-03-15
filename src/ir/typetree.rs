use chirrtl_parser::ast::*;
use std::fmt::{Debug, Display};
use std::collections::VecDeque;
use petgraph::{graph::{Graph, NodeIndex}, Direction::Outgoing};
use crate::common::graphviz::GraphViz;

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
pub enum TypeTreeNodeType {
    Ground,
    Fields,
    Array,
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct TypeTreeNode {
    pub name: Identifier,
    pub dir: Direction,
    pub tpe: TypeTreeNodeType,
}

impl TypeTreeNode {
    pub fn new(name: Identifier, dir: Direction, tpe: TypeTreeNodeType) -> Self {
        Self { name, dir, tpe }
    }
}

impl Display for TypeTreeNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let original = format!("{:?}", self);
        let clean_for_dot = original.replace('"', "");
        write!(f, "{}", clean_for_dot)
    }
}

type Tree = Graph<TypeTreeNode, u32>;

#[derive(Debug, Default, Clone)]
pub struct TypeTree {
    pub graph: Tree,
    pub root: Option<NodeIndex>
}

impl TypeTree {
    pub fn construct_tree(tpe: &Type, name: Identifier, dir: Direction) -> Self {
        let mut ret = Self::default();
        Self::construct_tree_recursive(tpe, name, dir, None, &mut ret);
        return ret;
    }

    fn create_node_and_add_to_parent(
        node_tpe: TypeTreeNodeType,
        name: Identifier,
        dir: Direction,
        parent_opt: Option<NodeIndex>,
        tree: &mut TypeTree
    ) -> NodeIndex {
        let child = tree.graph.add_node(TypeTreeNode::new(name, dir, node_tpe));
        match parent_opt {
            Some(parent) => {
                tree.graph.add_edge(parent, child, 0);
            }
            None => {
                tree.root = Some(child);
                // No parent, must have been root node
            }
        }
        child
    }

    fn construct_tree_recursive(
        cur_tpe: &Type,
        name: Identifier,
        dir: Direction,
        parent_opt: Option<NodeIndex>,
        tree: &mut Self
    ) {
        match cur_tpe {
            Type::TypeGround(_) => {
                Self::create_node_and_add_to_parent(
                    TypeTreeNodeType::Ground, name, dir, parent_opt, tree);
            }
            Type::TypeAggregate(ta) => {
                match ta.as_ref() {
                    TypeAggregate::Fields(fields) => {
                        let node_id = Self::create_node_and_add_to_parent(
                            TypeTreeNodeType::Fields, name, dir, parent_opt, tree);

                        for field in fields.iter() {
                            match field.as_ref() {
                                Field::Straight(name, tpe) => {
                                    Self::construct_tree_recursive(
                                        tpe, name.clone(), dir, Some(node_id), tree);
                                }
                                Field::Flipped(name, tpe) => {
                                    Self::construct_tree_recursive(
                                        tpe, name.clone(), dir.flip(), Some(node_id), tree);
                                }
                            }
                        }
                    }
                    TypeAggregate::Array(tpe, len) => {
                        let node_id = Self::create_node_and_add_to_parent(
                            TypeTreeNodeType::Array, name, dir, parent_opt, tree);

                        for id in 0..len.to_u32() {
                            Self::construct_tree_recursive(
                                tpe, Identifier::ID(Int::from(id)), dir, Some(node_id), tree);
                        }
                    }
                }
            }
            _ => {
                panic!("Unrecognized Type {:?}", cur_tpe);
            }
        }
    }

    pub fn all_possible_references(&self) -> Vec<Reference> {
        let mut ret = vec![];
        let mut q: VecDeque<(NodeIndex, Reference)> = VecDeque::new();
        if let Some(root_id) = self.root {
            let root = self.graph.node_weight(root_id).unwrap();
            q.push_back((root_id, Reference::Ref(root.name.clone())));
        }
        while !q.is_empty() {
            let (id, cur_ref) = q.pop_front().unwrap();
            ret.push(cur_ref.clone());

            let childs = self.graph.neighbors_directed(id, Outgoing);
            for child_id in childs {
                let child = self.graph.node_weight(child_id).unwrap();
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

    fn print_tree_recursive(&self, id: NodeIndex, depth: usize) {
        println!("{}{:?}", "  ".repeat(depth), self.graph.node_weight(id).unwrap());

        let childs = self.graph.neighbors_directed(id, Outgoing);
        for child in childs {
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

    fn ref_identifier_chain_recursive(reference: &Reference, chain: &mut VecDeque<Identifier>) {
        match reference {
            Reference::Ref(name) => {
                chain.push_back(name.clone());
            }
            Reference::RefDot(parent, child) => {
                Self::ref_identifier_chain_recursive(parent, chain);
                chain.push_back(child.clone());
            }
            Reference::RefIdxInt(parent, id) => {
                Self::ref_identifier_chain_recursive(parent, chain);
                chain.push_back(Identifier::ID(id.clone()));
            }
            Reference::RefIdxExpr(_parent, _addr) => {
                panic!("Reference identifier chain got recursive addressing {:?}", reference);
            }
        }
    }

    pub fn ref_identifier_chain(reference: &Reference) -> VecDeque<Identifier> {
        let mut ret = VecDeque::new();
        Self::ref_identifier_chain_recursive(reference, &mut ret);
        return ret;
    }


    pub fn subtree_root(&self, reference: &Reference) -> Option<NodeIndex> {
        let mut chain = Self::ref_identifier_chain(reference);
        let mut q: VecDeque<NodeIndex> = VecDeque::new();
        if let Some(root_id) = self.root {
            q.push_back(root_id);
        }

        let mut subtree_root_opt: Option<NodeIndex> = None;
        while !q.is_empty() && !chain.is_empty() {
            let id = q.pop_front().unwrap();
            subtree_root_opt = Some(id);

            let node = self.graph.node_weight(id).unwrap();
            let ref_name = chain.pop_front().unwrap();
            if node.name != ref_name {
                self.print_tree();
                panic!("Reference {:?} not inclusive in type tree", reference);
            }

            let childs = self.graph.neighbors_directed(id, Outgoing);
            for cid in childs {
                let child = self.graph.node_weight(cid).unwrap();
                if let Some(ref_name) = chain.front() {
                    if child.name == *ref_name {
                        q.push_back(cid);
                    }
                }
            }
        }

        assert!(chain.is_empty(), "Reference {:?} is not inclusive in type tree", reference);
        assert!(q.is_empty(),     "Queue {:?} still contains elements after finding subtree", q);

        return subtree_root_opt;

    }

    /// Given a `reference`, returns all the subtree leaf nodes
    /// ```
    ///       o
    ///     /  \
    ///    o   x
    ///   / \ / \
    ///   $ $ x x
    /// ```
    /// - If the matching reference path is `o`-`o`, return leaves marked `$`
    pub fn subtree_leaves(&self, reference: &Reference) -> Vec<NodeIndex> {
        let mut ret = vec![];
        let mut q: VecDeque<NodeIndex> = VecDeque::new();

        let subtree_root_opt = self.subtree_root(reference);
        if let Some(subtree_root) = subtree_root_opt {
            q.push_back(subtree_root);
        }

        while !q.is_empty() {
            let id = q.pop_front().unwrap();
            let childs = self.graph.neighbors_directed(id, Outgoing);
            let mut num_childs = 0;
            for cid in childs {
                num_childs += 1;
                q.push_back(cid);
            }
            if num_childs == 0 {
                ret.push(id);
            }
        }

        return ret;
    }
}

impl GraphViz<TypeTreeNode, u32> for TypeTree {
    fn node_indices(self: &Self) -> petgraph::graph::NodeIndices {
        self.graph.node_indices()
    }

    fn node_weight(self: &Self, id: NodeIndex) -> Option<&TypeTreeNode> {
        self.graph.node_weight(id)
    }

    fn edge_indices(self: &Self) -> petgraph::graph::EdgeIndices {
        self.graph.edge_indices()
    }

    fn edge_endpoints(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.graph.edge_endpoints(id)
    }

    fn edge_weight(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<&u32> {
        self.graph.edge_weight(id)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use chirrtl_parser::parse_circuit;

    #[test]
    fn print_type_tree() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./test-inputs/NestedBundle.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for port in m.ports.iter() {
                        let port_type_tree = match port.as_ref() {
                            Port::Input(name, tpe, _info) => {
                                TypeTree::construct_tree(tpe, name.clone(), Direction::Input)
                            }
                            Port::Output(name, tpe, _info) => {
                                TypeTree::construct_tree(tpe, name.clone(), Direction::Output)
                            }
                        };
                        port_type_tree.print_tree();
                    }

                }
                CircuitModule::ExtModule(_e) => {
                }
            }
        }
        return Ok(());
    }

    #[test]
    fn check_subtree_root() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./test-inputs/NestedBundle.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for port in m.ports.iter() {
                        match port.as_ref() {
                            Port::Output(name, tpe, _info) => {
                                let typetree = TypeTree::construct_tree(tpe, name.clone(), Direction::Output);
                                // let _ = typetree.export_graphviz("./test-outputs/NestedBundle.typetree.pdf", None, false);

                                let root = Reference::Ref(Identifier::Name("io".to_string()));
                                let subtree_root = typetree.subtree_root(&root);
                                assert_eq!(subtree_root, Some(NodeIndex::from(0)));

                                let g = Reference::RefDot(Box::new(root), Identifier::Name("g".to_string()));
                                let subtree_root = typetree.subtree_root(&g);
                                assert_eq!(subtree_root, Some(NodeIndex::from(1)));

                                let g1 = Reference::RefIdxInt(Box::new(g), Int::from(1));
                                let subtree_root = typetree.subtree_root(&g1);
                                assert_eq!(subtree_root, Some(NodeIndex::from(24)));

                                let g1f = Reference::RefDot(Box::new(g1), Identifier::Name("f".to_string()));
                                let subtree_root = typetree.subtree_root(&g1f);
                                assert_eq!(subtree_root, Some(NodeIndex::from(42)));

                                let subtree_leaves = typetree.subtree_leaves(&g1f);
                                assert_eq!(subtree_leaves, vec![NodeIndex::from(45), NodeIndex::from(44), NodeIndex::from(43)]);
                            }
                            _ => {
                            }
                        };
                    }

                }
                CircuitModule::ExtModule(_e) => {
                }
            }
        }
        return Ok(());
    }
}
