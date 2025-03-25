use chirrtl_parser::ast::*;
use std::fmt::{Debug, Display};
use std::collections::VecDeque;
use petgraph::{graph::{Graph, NodeIndex}, Direction::{Outgoing, Incoming}};
use petgraph::algo::is_isomorphic;
use ptree::{TreeItem, Style, write_tree, print_tree};
use indexmap::IndexMap;
use std::fs::File;
use std::io::BufWriter;
use std::hash::{Hash, Hasher};
use crate::common::graphviz::*;
use crate::common::RippleIRErr;
use crate::impl_clean_display;

/// - Direction in the perspective of the noding holding this `TypeTree`
/// ```
/// o <-- Incoming ---
/// o --- Outgoing -->
/// ```
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeDirection {
    #[default]
    Outgoing,
    Incoming,
}

impl TypeDirection {
    pub fn flip(&self) -> Self {
        match self {
            Self::Incoming => Self::Outgoing,
            Self::Outgoing => Self::Incoming,
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

/// Node in the TypeTree
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct TypeTreeNode {
    /// Name of this node. The root of the tree will not have a name
    pub name: Option<Identifier>,

    /// Direction of this node
    pub dir: TypeDirection,

    /// NodeType
    pub tpe: TypeTreeNodeType,

    /// Points to the Node in the flattened IR graph
    pub id: Option<NodeIndex>
}

impl TypeTreeNode {
    pub fn new(name: Option<Identifier>, dir: TypeDirection, tpe: TypeTreeNodeType) -> Self {
        Self { name, dir, tpe, id: None, }
    }
}

impl_clean_display!(TypeTreeNode);

/// Used for representing a path in the `TypeTree`
#[derive(Debug, Clone, Eq)]
pub struct TypeTreeNodePath {
    /// Direction of the type
    dir: TypeDirection,

    /// Type of this node
    tpe: TypeTreeNodeType,

    /// Reference representing the path in the `TypeTree`
    rc: Option<Reference>,
}

impl Hash for TypeTreeNodePath {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tpe.hash(state);
        self.rc.hash(state);
    }
}

impl PartialEq for TypeTreeNodePath {
    fn eq(&self, other: &Self) -> bool {
        self.rc == other.rc
    }
}

impl TypeTreeNodePath {
    pub fn new(dir: TypeDirection, tpe: TypeTreeNodeType, rc: Option<Reference>) -> Self {
        Self { dir, tpe, rc }
    }

    /// Add a `child` node to the path in the `TypeTree`
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

/// A tree that represents the type of an aggregate node
#[derive(Debug, Default, Clone)]
pub struct TypeTree {
    /// Graph representing the tree
    pub graph: Tree,

    /// NodeIndex of the root node
    pub root: Option<NodeIndex>
}

impl TypeTree {
    pub fn build_from_ground_type(gt: GroundType) -> Self {
        let mut ret = Self::default();
        let node = TypeTreeNode::new(None, TypeDirection::default(), TypeTreeNodeType::Ground(gt));
        let root = ret.graph.add_node(node);
        ret.root = Some(root);
        return ret;
    }

    /// Build a `TypeTree` that represents a `Type`
    pub fn build_from_type(tpe: &Type, dir: TypeDirection) -> Self {
        let mut ret = Self::default();
        ret.build_recursive(tpe, None, dir, None);
        return ret;
    }

    fn create_node_and_add_to_parent(
        &mut self,
        node_tpe: TypeTreeNodeType,
        name: Option<Identifier>,
        dir: TypeDirection,
        parent_opt: Option<NodeIndex>,
    ) -> NodeIndex {
        let child = self.graph.add_node(TypeTreeNode::new(name, dir, node_tpe));
        match parent_opt {
            Some(parent) => {
                self.graph.add_edge(parent, child, TypeTreeEdge::default());
            }
            None => {
                self.root = Some(child);
                // No parent, must have been root node
            }
        }
        child
    }

    fn build_recursive(
        &mut self,
        cur_tpe: &Type,
        name: Option<Identifier>,
        dir: TypeDirection,
        parent_opt: Option<NodeIndex>
    ) {
        match cur_tpe {
            Type::TypeGround(x) => {
                self.create_node_and_add_to_parent(
                    TypeTreeNodeType::Ground(GroundType::from(x)), name, dir, parent_opt);
            }
            Type::TypeAggregate(ta) => {
                match ta.as_ref() {
                    TypeAggregate::Fields(fields) => {
                        let node_id = self.create_node_and_add_to_parent(
                            TypeTreeNodeType::Fields, name, dir, parent_opt);

                        for field in fields.iter() {
                            match field.as_ref() {
                                Field::Straight(name, tpe) => {
                                    self.build_recursive(
                                        tpe, Some(name.clone()), dir, Some(node_id));
                                }
                                Field::Flipped(name, tpe) => {
                                    self.build_recursive(
                                        tpe, Some(name.clone()), dir.flip(), Some(node_id));
                                }
                            }
                        }
                    }
                    TypeAggregate::Array(tpe, len) => {
                        let node_id = self.create_node_and_add_to_parent(
                            TypeTreeNodeType::Array, name, dir, parent_opt);

                        for id in 0..len.to_u32() {
                            self.build_recursive(
                                tpe, Some(Identifier::ID(Int::from(id))), dir, Some(node_id));
                        }
                    }
                }
            }
            _ => {
                panic!("Unrecognized Type {:?}", cur_tpe);
            }
        }
    }

    /// Returns all possible references in this `TypeTree` instance
    /// ```
    ///     root
    ///   /     \
    ///   a     b
    ///        / \
    ///        c d
    /// ```
    /// - when `root_name` is `io`, and the typetree looks like the above,
    /// this function will return: `[ io, io.a, io.b, io.b.c, io.b.d ]`
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

    fn id_identifier_chain_recursive(&self, id: NodeIndex, chain: &mut String) {
        let parents = self.graph.neighbors_directed(id, Incoming);
        for pid in parents {
            self.id_identifier_chain_recursive(pid, chain);
        }

        let node = self.graph.node_weight(id).unwrap();

        let name = match &node.name {
            Some(x) => {
                match x {
                    Identifier::Name(y) => format!(".{}", y),
                    Identifier::ID(y) => format!("[{}]", y.to_u32())
                }
            }
            None => "".to_string()
        };

        match node.tpe {
            TypeTreeNodeType::Ground(..) => {
                chain.push_str(&name);
            }
            TypeTreeNodeType::Fields => {
                chain.push_str(&name);
            }
            TypeTreeNodeType::Array => {
                chain.push_str(&name);
            }
        }
    }

    /// Returns a stringified node name of a node in a tree
    pub fn node_name(&self, root: &Identifier, id: NodeIndex) -> Identifier {
        let mut ret = root.to_string();
        self.id_identifier_chain_recursive(id, &mut ret);
        return Identifier::Name(ret);
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
            Reference::RefIdxExpr(parent, _addr) => {
                // This is the same case as RefIdxInt except that
                // we are doing dynamic indexing
                Self::ref_identifier_chain_recursive(parent, chain);
                chain.push_back(Identifier::ID(Int::from(0)));
            }
        }
    }

    fn ref_identifier_chain(reference: &Reference) -> VecDeque<Identifier> {
        let mut ret = VecDeque::new();
        Self::ref_identifier_chain_recursive(reference, &mut ret);
        return ret;
    }

    /// Returns the root of the subtree given a `reference`
    /// ```
    ///     root
    ///   /     \
    ///   a     b
    ///        / \
    ///        c d
    /// ```
    /// - When `reference` is `io.b`, it will return NodeIndex for `b`
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

        if !chain.is_empty() {
            self.print_tree();
            panic!("Reference {:?} is not inclusive in type tree", reference);
        }
        if !q.is_empty() {
            self.print_tree();
            panic!("Queue {:?} still contains elements after finding subtree", q);
        }

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

    /// Given a `reference`, returns all the subtree leaf nodes along with the path to each node
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

            // Leaf node
            if num_childs == 0 {
                ret.insert(nirc.1, nirc.0);
            }
        }
        return ret;
    }

    /// Returns all leaf node indices of this tree
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

    /// Given `reference`, returns a subtree for this reference type
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

    /// Given `expr`, returns a subtree for this expr type
    pub fn subtree_from_expr(&self, expr: &Expr) -> Self {
        match expr {
            Expr::Reference(r) => {
                self.subtree_from_ref(r)
            }
            Expr::UIntInit(..)     |
                Expr::SIntInit(..) |
                Expr::PrimOp1Expr(..) => {
                self.clone()
            }
            _ => {
                panic!("Unexpected Expr on subtree_from_expr {:?}", expr);
            }
        }
    }

    pub fn subtree_array_element(&self) -> Self {
        let root = self.root.expect("to exist");
        assert_eq!(
            self.graph.node_weight(root).unwrap().tpe,
            TypeTreeNodeType::Array);

        self.subtree_from_ref(
            &Reference::RefIdxInt(
                Box::new(Reference::Ref(Identifier::Name("".to_string()))),
                Int::from(0)))
    }

    /// Checks if two typetrees are equivalent
    pub fn eq(&self, other: &Self) -> bool {
        is_isomorphic(&self.graph, &other.graph)
    }

    /// Given some TypeTrees with their names, create a new TypeTree that takes
    /// the given ones as a subtree
    pub fn merge_trees(ttrees: IndexMap<&Identifier, &Self>) -> Self {
        let mut ret = Self::default();

        let root_node = TypeTreeNode::new(None, TypeDirection::default(), TypeTreeNodeType::Fields);
        let root_id = ret.graph.add_node(root_node);
        ret.root = Some(root_id);

        for (name, ttree) in ttrees {
            let mut node_id_map: IndexMap<NodeIndex, NodeIndex> = IndexMap::new();

            // Add name to the subtree root
            let root_id = ttree.root.unwrap();
            let mut root = ttree.graph.node_weight(root_id).unwrap().clone();
            root.name = Some(name.clone());

            // Add the subtree root as a child to the root
            let new_id = ret.graph.add_node(root);
            ret.graph.add_edge(ret.root.unwrap(), new_id, TypeTreeEdge::default());
            node_id_map.insert(root_id, new_id);

            // Traverse the subtree and add all its childs
            let mut q: VecDeque<NodeIndex> = VecDeque::new();
            q.push_back(root_id);

            while !q.is_empty() {
                let id = q.pop_front().unwrap();
                let new_id = *node_id_map.get(&id).unwrap();

                let childs = ttree.graph.neighbors_directed(id, Outgoing);
                for cid in childs {
                    q.push_back(cid);

                    let child = ttree.graph.node_weight(cid).unwrap();
                    let new_child_id = ret.graph.add_node(child.clone());
                    ret.graph.add_edge(new_id, new_child_id, TypeTreeEdge::default());
                    node_id_map.insert(cid, new_child_id);
                }
            }
        }
        return ret;
    }

    /// Flips the directionality of all the TypeTreeNodes
    pub fn flip(&mut self) {
        for id in self.graph.node_indices() {
            let node = self.graph.node_weight_mut(id).unwrap();
            node.dir = node.dir.flip();
        }
    }
}

impl DefaultGraphVizCore<TypeTreeNode, TypeTreeEdge> for TypeTree {
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

impl GraphViz for TypeTree {
    fn graphviz_string(
        self: &Self,
        node_attr: Option<&NodeAttributeMap>,
        edge_attr: Option<&EdgeAttributeMap>
    ) -> Result<String, std::io::Error> {
        DefaultGraphVizCore::graphviz_string(self, node_attr, edge_attr)
    }
}

/// Helper struct for printing the `TypeTree`
#[derive(Clone)]
pub struct TypeTreePrinter<'a> {
    pub tree: &'a Graph<TypeTreeNode, TypeTreeEdge>,
    pub idx: NodeIndex,
}

impl<'a> TypeTreePrinter<'a> {
    pub fn new(tree: &'a Graph<TypeTreeNode, TypeTreeEdge>, idx: NodeIndex) -> Self {
        Self {
            tree,
            idx,
        }
    }

    pub fn write_to_file(&self, file: &str) -> Result<(), RippleIRErr> {
        let file = File::create(file)?;
        let writer = BufWriter::new(file);
        write_tree(self, writer)?;
        Ok(())
    }

    pub fn print(&self) -> Result<(), RippleIRErr> {
        print_tree(self)?;
        Ok(())
    }
}

impl<'a> TreeItem for TypeTreePrinter<'a> {
    type Child = Self;

    fn write_self<W: std::io::Write>(&self, f: &mut W, style: &Style) -> std::io::Result<()> {
        if let Some(w) = self.tree.node_weight(self.idx) {
            write!(f, "{} {:?}", style.paint(w), self.idx)
        } else {
            Ok(())
        }
    }

    fn children(&self) -> std::borrow::Cow<[Self::Child]> {
        let v: Vec<_> = self.tree
            .neighbors_directed(self.idx, Outgoing)
            .map(|i| TypeTreePrinter::new(self.tree, i))
            .collect();
        let v_rev: Vec<_> = v.iter().map(|x| x.clone()).rev().collect();
        std::borrow::Cow::from(v_rev)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::common::RippleIRErr;
    use chirrtl_parser::parse_circuit;

    #[test]
    fn check_subtree_array_element() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/SinglePortSRAM.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for stmt in &m.stmts {
                        match stmt.as_ref() {
                            Stmt::ChirrtlMemory(mem) => {
                                match mem {
                                    ChirrtlMemory::SMem(_, tpe, ..) => {
                                        let ttree = TypeTree::build_from_type(tpe, TypeDirection::Outgoing);
                                        let ttree_ae = ttree.subtree_array_element();

                                        let type_agg = TypeAggregate::Array(
                                            Box::new(Type::TypeGround(TypeGround::UInt(Some(Width(2))))),
                                            Int::from(4));
                                        let expected_type = Type::TypeAggregate(Box::new(type_agg));
                                        let ttree_ae_expected = TypeTree::build_from_type(&expected_type, TypeDirection::Outgoing);
                                        assert!(ttree_ae.eq(&ttree_ae_expected));
                                    }
                                    _ => { }
                                }
                            }
                            _ => { }
                        }
                    }
                }
                _ => { }
            }
        }
        Ok(())
    }

    #[test]
    fn check_gcd_name() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/GCD.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for port in m.ports.iter() {
                        match port.as_ref() {
                            Port::Output(_name, tpe, _info) => {
                                let tt = TypeTree::build_from_type(tpe, TypeDirection::Outgoing);
                                assert_eq!(tt.node_name(&Identifier::Name("io".to_string()), NodeIndex::from(1)).to_string(), "io.value1".to_string());
                                assert_eq!(tt.node_name(&Identifier::Name("io".to_string()), NodeIndex::from(2)).to_string(), "io.value2".to_string());
                                assert_eq!(tt.node_name(&Identifier::Name("io".to_string()), NodeIndex::from(3)).to_string(), "io.loadingValues".to_string());
                                assert_eq!(tt.node_name(&Identifier::Name("io".to_string()), NodeIndex::from(4)).to_string(), "io.outputGCD".to_string());
                                assert_eq!(tt.node_name(&Identifier::Name("io".to_string()), NodeIndex::from(5)).to_string(), "io.outputValid".to_string());
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
        Ok(())
    }

    #[test]
    fn print_type_tree() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/NestedBundle.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for port in m.ports.iter() {
                        let port_type_tree = match port.as_ref() {
                            Port::Input(_name, tpe, _info) => {
                                TypeTree::build_from_type(tpe, TypeDirection::Incoming)
                            }
                            Port::Output(_name, tpe, _info) => {
                                TypeTree::build_from_type(tpe, TypeDirection::Outgoing)
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
    fn check_subtree_root() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/NestedBundle.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for port in m.ports.iter() {
                        match port.as_ref() {
                            Port::Output(_name, tpe, _info) => {
                                let typetree = TypeTree::build_from_type(tpe, TypeDirection::Outgoing);
                                let _ = typetree.export_graphviz("./test-outputs/NestedBundle.typetree.pdf", None, None, false);

                                let root = Reference::Ref(Identifier::Name("io".to_string()));
                                let subtree_root = typetree.subtree_root(&root);
                                assert_eq!(subtree_root, Some(NodeIndex::from(0)));

                                let g = Reference::RefDot(Box::new(root), Identifier::Name("g".to_string()));
                                let subtree_root = typetree.subtree_root(&g);
                                assert_eq!(subtree_root, Some(NodeIndex::from(1)));

                                let g_name = typetree.node_name(&Identifier::Name("io".to_string()), subtree_root.unwrap());
                                assert_eq!(g_name.to_string(), "io.g".to_string());

                                let g1 = Reference::RefIdxInt(Box::new(g), Int::from(1));
                                let subtree_root = typetree.subtree_root(&g1);
                                assert_eq!(subtree_root, Some(NodeIndex::from(24)));

                                let g1_name = typetree.node_name(&Identifier::Name("io".to_string()), subtree_root.unwrap());
                                assert_eq!(g1_name.to_string(), "io.g[1]".to_string());

                                let g1f = Reference::RefDot(Box::new(g1), Identifier::Name("f".to_string()));
                                let subtree_root = typetree.subtree_root(&g1f);
                                assert_eq!(subtree_root, Some(NodeIndex::from(42)));

                                let g1f_name = typetree.node_name(&Identifier::Name("io".to_string()), subtree_root.unwrap());
                                assert_eq!(g1f_name.to_string(), "io.g[1].f".to_string());

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

    #[test]
    fn check_subtree_leaves_with_path() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/NestedBundle.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for port in m.ports.iter() {
                        match port.as_ref() {
                            Port::Output(_name, tpe, _info) => {
                                let typetree = TypeTree::build_from_type(tpe, TypeDirection::Outgoing);

                                let _ = typetree.export_graphviz("./test-outputs/NestedBundle.typetree.pdf", None, None, false);

                                let root = Reference::Ref(Identifier::Name("io".to_string()));
                                let g = Reference::RefDot(Box::new(root), Identifier::Name("g".to_string()));
                                let g1 = Reference::RefIdxInt(Box::new(g), Int::from(1));
                                let g1f = Reference::RefDot(Box::new(g1.clone()), Identifier::Name("f".to_string()));
                                let leaves_with_path = typetree.subtree_leaves_with_path(&g1f);

                                let mut expect: IndexMap<TypeTreeNodePath, NodeIndex> = IndexMap::new();
                                expect.insert(
                                    TypeTreeNodePath::new(
                                        TypeDirection::Outgoing,
                                        TypeTreeNodeType::Ground(GroundType::UInt),
                                        Some(Reference::Ref(Identifier::ID(Int::from(2))))),
                                    NodeIndex::from(45));

                                expect.insert(
                                    TypeTreeNodePath::new(
                                        TypeDirection::Outgoing,
                                        TypeTreeNodeType::Ground(GroundType::UInt),
                                        Some(Reference::Ref(Identifier::ID(Int::from(1))))),
                                    NodeIndex::from(44));

                                expect.insert(
                                    TypeTreeNodePath::new(
                                        TypeDirection::Outgoing,
                                        TypeTreeNodeType::Ground(GroundType::UInt),
                                        Some(Reference::Ref(Identifier::ID(Int::from(0))))),
                                    NodeIndex::from(43));
                                assert_eq!(leaves_with_path, expect);
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

    #[test]
    fn check_subtree_from_expr() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/DecoupledMux.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        for module in circuit.modules.iter() {
            match module.as_ref() {
                CircuitModule::Module(m) => {
                    for port in m.ports.iter() {
                        match port.as_ref() {
                            Port::Output(_name, tpe, _info) => {
                                let typetree = TypeTree::build_from_type(tpe, TypeDirection::Outgoing);

                                let _ = typetree.export_graphviz("./test-outputs/DecoupledMux.typetree.pdf", None, None, false);

                                let root = Reference::Ref(Identifier::Name("io".to_string()));
                                let c = Reference::RefDot(Box::new(root), Identifier::Name("c".to_string()));
                                let c_ready = Reference::RefDot(Box::new(c.clone()), Identifier::Name("ready".to_string()));

                                let subtree = typetree.subtree_from_expr(&Expr::Reference(c_ready));
                                let node = subtree.graph.node_weight(NodeIndex::from(0)).unwrap();
                                assert_eq!(*node,
                                    TypeTreeNode::new(
                                        None,
                                        TypeDirection::Incoming,
                                        TypeTreeNodeType::Ground(GroundType::UInt)));
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
