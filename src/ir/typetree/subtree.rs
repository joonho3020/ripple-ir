use crate::ir::typetree::typetree::*;
use chirrtl_parser::ast::*;
use petgraph::graph::Neighbors;
use petgraph::{graph::NodeIndex, Direction::{Outgoing, Incoming}};
use std::collections::VecDeque;
use indexmap::IndexMap;

#[derive(Debug)]
pub struct SubTreeView<'a> {
    ttree: &'a TypeTree,
    root: NodeIndex
}

impl<'a> SubTreeView<'a> {
    pub fn new(ttree: &'a TypeTree, root: TTreeNodeIndex) -> Self {
        Self { ttree, root }
    }

    pub fn from_subtree(subtree: &'a SubTreeView<'a>, root: TTreeNodeIndex) -> Self {
        Self { ttree: subtree.ttree(), root }
    }

    pub fn root_node(&self) -> Option<TypeTreeNode> {
        let mut node = self.ttree.graph.node_weight(self.root).unwrap().clone();
        node.name = None;
        Some(node)
    }

    pub fn get_node(&self, id: TTreeNodeIndex) -> Option<TypeTreeNode> {
        if id == self.root {
            self.root_node()
        } else {
            Some(self.ttree.graph.node_weight(id).unwrap().clone())
        }
    }

    pub fn childs(&self, id: TTreeNodeIndex) -> Neighbors<TypeTreeEdge> {
        self.ttree.graph.neighbors_directed(id, Outgoing)
    }

    pub fn parents(&self, id: TTreeNodeIndex) -> Neighbors<TypeTreeEdge> {
        self.ttree.graph.neighbors_directed(id, Incoming)
    }

    pub fn is_ground_type(&self) -> bool {
        match self.root_node().unwrap().tpe {
            TypeTreeNodeType::Ground(..) => { true }
            _ => { false }
        }
    }

    pub fn ttree(&self) -> &'a TypeTree {
        self.ttree
    }

    /// Clones a `SubTreeView` into a `TypeTree`
    /// - TTreeNodeIndex is not preserved between this instance and the cloned child. Must be careful
    /// when using this API around stuff that contains metadata about TTreeNodeIndex of `TypeTree`s
    pub fn clone_ttree(&self) -> TypeTree {
        type NewId = TTreeNodeIndex;
        type OldId = TTreeNodeIndex;
        type IdTuple = (OldId, Option<NewId>);

        let mut q: VecDeque<IdTuple> = VecDeque::new();
        q.push_back((self.root, None));

        let mut ret = TypeTree::default();
        while !q.is_empty() {
            let id_tuple = q.pop_front().unwrap();
            let node = self.get_node(id_tuple.0).unwrap();

            let new_id = match id_tuple.1 {
                Some(parent_id) => {
                    let sid = ret.graph.add_node(node);
                    ret.graph.add_edge(parent_id, sid, TypeTreeEdge::default());
                    sid
                }
                None => {
                    let sid = ret.graph.add_node(node);
                    ret.root = Some(sid);
                    sid
                }
            };

            let childs = self.childs(id_tuple.0);
            for child in childs {
                q.push_back((child, Some(new_id)));
            }
        }
        return ret;
    }

    fn print_tree_recursive(&self, id: TTreeNodeIndex, depth: usize) {
        println!("{}{:?}", "  ".repeat(depth), self.get_node(id).unwrap());

        for child in self.childs(id) {
            self.print_tree_recursive(child, depth + 1);
        }
    }

    pub fn print_tree(&self) {
        self.print_tree_recursive(self.root, 0);
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
        let mut q: VecDeque<(TTreeNodeIndex, Reference)> = VecDeque::new();
        q.push_back((self.root, Reference::Ref(root_name)));

        while !q.is_empty() {
            let (id, cur_ref) = q.pop_front().unwrap();
            ret.push(cur_ref.clone());

            for cid in self.childs(id) {
                let child = self.get_node(cid).unwrap();
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
                q.push_back((cid, child_ref));
            }
        }
        return ret;
    }

    fn id_identifier_chain_recursive(&self, id: TTreeNodeIndex, chain: &mut String) {
        for pid in self.parents(id) {
            self.id_identifier_chain_recursive(pid, chain);
        }

        let node = self.get_node(id).unwrap();
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
    pub fn node_name(&self, root_name: &Identifier, id: TTreeNodeIndex) -> Identifier {
        let mut ret = root_name.to_string();
        self.id_identifier_chain_recursive(id, &mut ret);
        return Identifier::Name(ret);
    }


    /// Returns all leaf node indices of this subtree
    pub fn leaves(&self) -> Vec<TTreeNodeIndex> {
        let mut ret = vec![];
        let mut q: VecDeque<TTreeNodeIndex> = VecDeque::new();
        q.push_back(self.root);

        while !q.is_empty() {
            let id = q.pop_front().unwrap();
            let childs = self.childs(id);
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

    /// Given a `reference`, returns all the subtree leaf nodes
    /// ```
    ///       o
    ///     /  \
    ///    o   x
    ///   / \ / \
    ///   $ $ x x
    /// ```
    /// - If the matching reference path is `o`-`o`, return leaves marked `$`
    pub fn subtree_leaves(&self, reference: &Reference) -> Vec<TTreeNodeIndex> {
        let subtree_root_opt = self.subtree_root(reference);
        if let Some(subtree_root) = subtree_root_opt {
            let subtree = SubTreeView::from_subtree(self, subtree_root);
            subtree.leaves()
        } else {
            vec![]
        }
    }

    pub fn leaves_with_path(&self) -> IndexMap<TypeTreeNodePath, TTreeNodeIndex> {
        let mut ret = IndexMap::new();

        let mut q: VecDeque<(TTreeNodeIndex, TypeTreeNodePath)> = VecDeque::new();
        let n = self.root_node().unwrap();

        q.push_back((self.root, TypeTreeNodePath::new(n.dir, n.tpe.clone(), None)));

        while !q.is_empty() {
            let nirc = q.pop_front().unwrap();

            let mut num_childs = 0;
            for cid in self.childs(nirc.0) {
                let cnode = self.get_node(cid).unwrap();
                num_childs += 1;

                let mut path = nirc.1.clone();
                path.append(&cnode);
                q.push_back((cid, path));
            }

            // Leaf node
            if num_childs == 0 {
                ret.insert(nirc.1, nirc.0);
            }
        }
        return ret;
    }

    /// Given a `reference`, returns all the subtree leaf nodes along with the path to each node
    pub fn subtree_leaves_with_path(&self, reference: &Reference) -> IndexMap<TypeTreeNodePath, TTreeNodeIndex> {
        let subtree_root_opt = self.subtree_root(reference);
        if let Some(subtree_root) = subtree_root_opt {
            let subtree = SubTreeView::from_subtree(self, subtree_root);
            subtree.leaves_with_path()
        } else {
            IndexMap::new()
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
    /// - When `reference` is `io.b`, it will return TTreeNodeIndex for `b`
    pub fn subtree_root(&self, reference: &Reference) -> Option<TTreeNodeIndex> {
        let mut chain = Self::ref_identifier_chain(reference);
        let mut q: VecDeque<TTreeNodeIndex> = VecDeque::new();
        q.push_back(self.root);

        let mut subtree_root_opt: Option<TTreeNodeIndex> = None;
        while !q.is_empty() && !chain.is_empty() {
            let id = q.pop_front().unwrap();
            subtree_root_opt = Some(id);

            let node = self.get_node(id).unwrap();
            let ref_name = chain.pop_front().unwrap();
            if id != self.root && *node.name.as_ref().unwrap() != ref_name {
                self.print_tree();
                panic!("Reference {:?} not inclusive in type tree", reference);
            }

            for cid in self.childs(id) {
                let child = self.get_node(cid).unwrap();
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

    /// Given `reference`, returns a subtree for this reference type
    pub fn subtree_from_ref(&self, reference: &Reference) -> Option<SubTreeView<'_>> {
        let subtree_root_opt = self.subtree_root(reference);
        if let Some(subtree_root) = subtree_root_opt {
            Some(SubTreeView::from_subtree(self, subtree_root))
        } else {
            None
        }
    }

    /// Given `expr`, returns a subtree for this expr type
    pub fn subtree_from_expr(&self, expr: &Expr) -> Option<SubTreeView<'_>> {
        match expr {
            Expr::Reference(r) => {
                self.subtree_from_ref(r)
            }
            Expr::UIntInit(..) |
            Expr::SIntInit(..) |
            Expr::PrimOp1Expr(..) => {
                Some(SubTreeView::from_subtree(self, self.root))
            }
            _ => {
                panic!("Unexpected Expr on subtree_from_expr {:?}", expr);
            }
        }
    }

    pub fn subtree_array_element(&self) -> SubTreeView<'_> {
        assert_eq!(self.root_node().unwrap().tpe, TypeTreeNodeType::Array);
        let elem_subtree = self.subtree_from_ref(
            &Reference::RefIdxInt(
                Box::new(Reference::Ref(Identifier::Name("".to_string()))),
                Int::from(0))).expect("to exist");
        return elem_subtree;
    }

    pub fn eq(&self, other: &SubTreeView) -> bool {
        self.eq_recursive(self.root, other, other.root)
    }

    fn eq_recursive(&self, id: TTreeNodeIndex, other: &SubTreeView, other_id: TTreeNodeIndex) -> bool {
        let node_opt = self.get_node(id);
        let other_node_opt = other.get_node(other_id);

        match (node_opt, other_node_opt) {
            (Some(node), Some(other_node)) => {
                if node != other_node {
                    false
                } else {
                    let mut childs: Vec<TTreeNodeIndex> = self.childs(id).collect();
                    let mut other_childs: Vec<TTreeNodeIndex> = other.childs(other_id).collect();
                    if childs.len() != other_childs.len() {
                        false
                    } else {
                        childs.sort_by_key(|&i| self.get_node(i).unwrap());
                        other_childs.sort_by_key(|&i| other.get_node(i).unwrap());

                        for (s_child, o_child) in childs.into_iter().zip(other_childs.into_iter()) {
                            if !self.eq_recursive(s_child, other, o_child) {
                                return false;
                            }
                        }
                        true
                    }
                }
            }
            _ => {
                false
            }
        }
    }
}

