use chirrtl_parser::ast::*;
use std::fmt::{Debug, Display};
use std::collections::VecDeque;
use std::process::Output;
use petgraph::{graph::{Graph, NodeIndex}, Direction::Outgoing};
use indexmap::IndexMap;
use crate::common::graphviz::GraphViz;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GroundType {
    Invalid,
    DontCare,
    Clock,
    Reset,
    AsyncReset,
    UInt,
    SInt,
    SMem,
    CMem,
    Inst,
}

impl From<&TypeGround> for GroundType {
    fn from(value: &TypeGround) -> Self {
        match value {
            TypeGround::SInt(..) => Self::SInt,
            TypeGround::UInt(..) => Self::UInt,
            TypeGround::Clock => Self::Clock,
            TypeGround::Reset => Self::Reset,
            TypeGround::AsyncReset => Self::AsyncReset,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeTreeNodeType {
    Ground(GroundType),
    Fields,
    Array,
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct TypeTreeNode {
    pub name: Option<Identifier>,
    pub dir: Direction,
    pub tpe: TypeTreeNodeType,
    pub id: Option<NodeIndex>
}

impl TypeTreeNode {
    pub fn new(name: Option<Identifier>, dir: Direction, tpe: TypeTreeNodeType) -> Self {
        Self { name, dir, tpe, id: None, }
    }
}

impl Display for TypeTreeNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let original = format!("{:?}", self);
        let clean_for_dot = original.replace('"', "");
        write!(f, "{}", clean_for_dot)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TypeTreeNodePath {
    dir: Direction,
    tpe: TypeTreeNodeType,
    rc:  Option<Reference>,
}

impl TypeTreeNodePath {
    pub fn new(dir: Direction, tpe: TypeTreeNodeType, rc: Option<Reference>) -> Self {
        Self { dir, tpe, rc }
    }

    pub fn append(&mut self, child: &TypeTreeNode) {
        self.dir = child.dir;
        self.rc = match &self.rc {
            None => Some(Reference::Ref(child.name.clone().unwrap())),
            Some(par) => {
                match self.tpe {
                    TypeTreeNodeType::Ground(_) => Some(Reference::Ref(child.name.clone().unwrap())),
                    TypeTreeNodeType::Fields    => Some(Reference::RefDot(Box::new(par.clone()), child.name.clone().unwrap())),
                    TypeTreeNodeType::Array     => {
                        let id = match &child.name {
                            Some(Identifier::ID(i)) => i,
                            _ => panic!("Array type should have ID as child, got {:?}", child.name)
                        };
                        Some(Reference::RefIdxInt(Box::new(par.clone()), id.clone()))
                    }
                }
            }
        };
        self.tpe = child.tpe.clone();
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct TypeTreeEdge(u32);

impl Display for TypeTreeEdge {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

type Tree = Graph<TypeTreeNode, TypeTreeEdge>;

#[derive(Debug, Default, Clone)]
pub struct TypeTree {
    pub graph: Tree,
    pub root: Option<NodeIndex>
}

impl TypeTree {
    pub fn construct_tree_from_ground_type(gt: GroundType) -> Self {
        let mut ret = Self::default();
        let node = TypeTreeNode::new(None, Direction::Output, TypeTreeNodeType::Ground(gt));
        let root = ret.graph.add_node(node);
        ret.root = Some(root);
        return ret;
    }

    pub fn construct_tree(tpe: &Type, dir: Direction) -> Self {
        let mut ret = Self::default();
        Self::construct_tree_recursive(tpe, None, dir, None, &mut ret);
        return ret;
    }

    fn create_node_and_add_to_parent(
        node_tpe: TypeTreeNodeType,
        name: Option<Identifier>,
        dir: Direction,
        parent_opt: Option<NodeIndex>,
        tree: &mut TypeTree
    ) -> NodeIndex {
        let child = tree.graph.add_node(TypeTreeNode::new(name, dir, node_tpe));
        match parent_opt {
            Some(parent) => {
                tree.graph.add_edge(parent, child, TypeTreeEdge::default());
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
        name: Option<Identifier>,
        dir: Direction,
        parent_opt: Option<NodeIndex>,
        tree: &mut Self
    ) {
        match cur_tpe {
            Type::TypeGround(x) => {
                Self::create_node_and_add_to_parent(
                    TypeTreeNodeType::Ground(GroundType::from(x)), name, dir, parent_opt, tree);
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
                                        tpe, Some(name.clone()), dir, Some(node_id), tree);
                                }
                                Field::Flipped(name, tpe) => {
                                    Self::construct_tree_recursive(
                                        tpe, Some(name.clone()), dir.flip(), Some(node_id), tree);
                                }
                            }
                        }
                    }
                    TypeAggregate::Array(tpe, len) => {
                        let node_id = Self::create_node_and_add_to_parent(
                            TypeTreeNodeType::Array, name, dir, parent_opt, tree);

                        for id in 0..len.to_u32() {
                            Self::construct_tree_recursive(
                                tpe, Some(Identifier::ID(Int::from(id))), dir, Some(node_id), tree);
                        }
                    }
                }
            }
            _ => {
                panic!("Unrecognized Type {:?}", cur_tpe);
            }
        }
    }

    pub fn all_possible_references(&self, root_name: Identifier) -> Vec<Reference> {
        let mut ret = vec![];
        let mut q: VecDeque<(NodeIndex, Reference)> = VecDeque::new();
        if let Some(root_id) = self.root {
            q.push_back((root_id, Reference::Ref(root_name)));
        }

        while !q.is_empty() {
            let (id, cur_ref) = q.pop_front().unwrap();
            ret.push(cur_ref.clone());

            let childs = self.graph.neighbors_directed(id, Outgoing);
            for child_id in childs {
                let child = self.graph.node_weight(child_id).unwrap();
                let child_ref = match &child.name {
                    Some(Identifier::ID(id)) => {
                        Reference::RefIdxInt(Box::new(cur_ref.clone()), id.clone())
                    }
                    Some(Identifier::Name(_)) => {
                        Reference::RefDot(Box::new(cur_ref.clone()), child.name.clone().unwrap())
                    }
                    _ => {
                        panic!("Child nodes in the tree must have a name {:?}", child);
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

    fn ref_identifier_chain(reference: &Reference) -> VecDeque<Identifier> {
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
            if id != self.root.unwrap() && *node.name.as_ref().unwrap() != ref_name {
                self.print_tree();
                panic!("Reference {:?} not inclusive in type tree", reference);
            }

            let childs = self.graph.neighbors_directed(id, Outgoing);
            for cid in childs {
                let child = self.graph.node_weight(cid).unwrap();
                if let Some(ref_name) = chain.front() {
                    if *child.name.as_ref().unwrap() == *ref_name {
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

    pub fn subtree_leaves_with_path(&self, reference: &Reference) -> IndexMap<TypeTreeNodePath, NodeIndex> {
        let mut ret = IndexMap::new();

        let mut q: VecDeque<(NodeIndex, TypeTreeNodePath)> = VecDeque::new();
        let subtree_root_opt = self.subtree_root(reference);
        if let Some(subtree_root) = subtree_root_opt {
            let n = self.graph.node_weight(subtree_root).unwrap();
            q.push_back((subtree_root, TypeTreeNodePath::new(n.dir, n.tpe.clone(), None)));
        }

        while !q.is_empty() {
            let nirc = q.pop_front().unwrap();
            let childs = self.graph.neighbors_directed(nirc.0, Outgoing);

            let mut num_childs = 0;
            for cid in childs {
                let cnode = self.graph.node_weight(cid).unwrap();
                num_childs += 1;

                let mut path = nirc.1.clone();
                path.append(cnode);
                q.push_back((cid, path));
            }

            if num_childs == 0 {
                ret.insert(nirc.1, nirc.0);
            }
        }

        return ret;
    }

    pub fn all_leaves(&self) -> Vec<NodeIndex> {
        let mut ret = vec![];
        let mut q: VecDeque<NodeIndex> = VecDeque::new();

        if let Some(root) = self.root {
            q.push_back(root);
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

    pub fn subtree_from_ref(&self, reference: &Reference) -> Self {
        let subtree_root_opt = self.subtree_root(reference);

        type NewId = NodeIndex;
        type OldId = NodeIndex;
        type IdTuple = (OldId, Option<NewId>);

        let mut q: VecDeque<IdTuple> = VecDeque::new();
        if let Some(subtree_root) = subtree_root_opt {
            q.push_back((subtree_root, None));
        }

        let mut ret = Self::default();
        while !q.is_empty() {
            let id_tuple = q.pop_front().unwrap();
            let mut node = self.graph.node_weight(id_tuple.0).unwrap().clone();

            let new_id = match id_tuple.1 {
                Some(parent_id) => {
                    let sid = ret.graph.add_node(node);
                    ret.graph.add_edge(parent_id, sid, TypeTreeEdge::default());
                    sid
                }
                None => {
                    node.name = None;
                    let sid = ret.graph.add_node(node);
                    ret.root = Some(sid);
                    sid
                }
            };

            let childs = self.graph.neighbors_directed(id_tuple.0, Outgoing);
            for child in childs {
                q.push_back((child, Some(new_id)));
            }
        }
        return ret;
    }
}

impl GraphViz<TypeTreeNode, TypeTreeEdge> for TypeTree {
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

    fn edge_weight(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<&TypeTreeEdge> {
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
                            Port::Input(_name, tpe, _info) => {
                                TypeTree::construct_tree(tpe, Direction::Input)
                            }
                            Port::Output(_name, tpe, _info) => {
                                TypeTree::construct_tree(tpe, Direction::Output)
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
                            Port::Output(_name, tpe, _info) => {
                                let typetree = TypeTree::construct_tree(tpe, Direction::Output);
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
