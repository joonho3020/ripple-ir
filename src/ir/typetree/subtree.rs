use crate::ir::typetree::typetree::*;
use crate::ir::typetree::tnode::*;
use crate::ir::typetree::path::*;
use crate::ir::typetree::tedge::*;
use chirrtl_parser::ast::*;
use petgraph::graph::Neighbors;
use petgraph::{graph::NodeIndex, Direction::{Outgoing, Incoming}};
use std::collections::VecDeque;
use indexmap::IndexMap;

pub type LeavesWithPath = IndexMap<TypeTreeNodePath, TypeTreeNodeIndex>;

#[derive(Debug)]
pub struct SubTreeView<'a> {
    ttree: &'a TypeTree,
    root: NodeIndex
}

impl<'a> SubTreeView<'a> {
    pub fn new(ttree: &'a TypeTree, root: NodeIndex) -> Self {
        Self { ttree, root }
    }

    pub fn from_subtree(subtree: &'a SubTreeView<'a>, root: NodeIndex) -> Self {
        Self { ttree: subtree.ttree(), root }
    }

    pub fn root_node(&self) -> Option<TypeTreeNode> {
        let mut node = self.ttree.graph.node_weight(self.root).unwrap().clone();
        node.name = None;
        Some(node)
    }

    pub fn is_empty(&self) -> bool {
        let root_uid = self.ttree.unique_id(self.root).unwrap();
        let child_cnt =  self.childs(*root_uid).count();
        if child_cnt == 0 && (self.root_node().unwrap().tpe == TypeTreeNodeType::Fields) {
            true
        } else {
            false
        }
    }

    pub fn is_array(&self) -> bool {
        self.root_node().unwrap().tpe == TypeTreeNodeType::Array
    }

    pub fn get_node(&self, id: TypeTreeNodeIndex) -> Option<TypeTreeNode> {
        let graph_id = *self.ttree.graph_id(id).unwrap();
        if graph_id == self.root {
            self.root_node()
        } else {
            Some(self.ttree.graph.node_weight(graph_id).unwrap().clone())
        }
    }

    pub fn childs(&self, id: TypeTreeNodeIndex) -> Neighbors<TypeTreeEdge> {
        let graph_id = *self.ttree.graph_id(id).unwrap();
        self.ttree.graph.neighbors_directed(graph_id, Outgoing)
    }

    pub fn parents(&self, id: TypeTreeNodeIndex) -> Neighbors<TypeTreeEdge> {
        let graph_id = *self.ttree.graph_id(id).unwrap();
        self.ttree.graph.neighbors_directed(graph_id, Incoming)
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
    /// - NodeIndex is not preserved between this instance and the cloned child. Must be careful
    /// when using this API around stuff that contains metadata about NodeIndex of `TypeTree`s
    pub fn clone_ttree(&self) -> TypeTree {
        type NewId = NodeIndex;
        type OldId = NodeIndex;
        type IdTuple = (OldId, Option<NewId>);

        let mut q: VecDeque<IdTuple> = VecDeque::new();
        q.push_back((self.root, None));

        let mut ret = TypeTree::default();
        while !q.is_empty() {
            let id_tuple = q.pop_front().unwrap();
            let unique_id = self.ttree.unique_id(id_tuple.0).unwrap();
            let node = self.get_node(*unique_id).unwrap();

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

            let childs = self.childs(*unique_id);
            for child in childs {
                q.push_back((child, Some(new_id)));
            }
        }
        ret.assign_unique_id();
        return ret;
    }

    fn print_tree_recursive(&self, id: NodeIndex, depth: usize) {
        let unique_id = *self.ttree.unique_id(id).unwrap();
        println!("{}{:?}", "  ".repeat(depth), self.get_node(unique_id).unwrap());

        for child in self.childs(unique_id) {
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
        let mut q: VecDeque<(TypeTreeNodeIndex, Reference)> = VecDeque::new();

        let unique_root = self.ttree.unique_id(self.root).unwrap();
        q.push_back((*unique_root, Reference::Ref(root_name)));

        while !q.is_empty() {
            let (id, cur_ref) = q.pop_front().unwrap();
            ret.push(cur_ref.clone());
            for cid in self.childs(id) {
                let unique_cid = *self.ttree.unique_id(cid).unwrap();
                let child = self.get_node(unique_cid).unwrap();
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
                q.push_back((unique_cid, child_ref));
            }
        }
        return ret;
    }

    fn id_identifier_chain_recursive(&self, id: TypeTreeNodeIndex, chain: &mut String) {
        for pid in self.parents(id) {
            let unique_pid = self.ttree.unique_id(pid).unwrap();
            self.id_identifier_chain_recursive(*unique_pid, chain);
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
    pub fn node_name(&self, root_name: &Identifier, id: TypeTreeNodeIndex) -> Identifier {
        let mut ret = root_name.to_string();
        self.id_identifier_chain_recursive(id, &mut ret);
        return Identifier::Name(ret);
    }

    pub fn all_ids(&self) -> Vec<TypeTreeNodeIndex> {
        let mut ret = vec![];
        let mut q: VecDeque<TypeTreeNodeIndex> = VecDeque::new();
        let unique_root = *self.ttree.unique_id(self.root).unwrap();
        q.push_back(unique_root);

        while !q.is_empty() {
            let id = q.pop_front().unwrap();
            let childs = self.childs(id);
            for cid in childs {
                let unique_cid = *self.ttree.unique_id(cid).unwrap();
                q.push_back(unique_cid);
            }
            ret.push(id);
        }
        return ret;
    }

    /// Returns all leaf node indices of this subtree
    pub fn leaves(&self) -> Vec<TypeTreeNodeIndex> {
        let mut ret = vec![];
        let mut q: VecDeque<TypeTreeNodeIndex> = VecDeque::new();
        let unique_root = *self.ttree.unique_id(self.root).unwrap();
        q.push_back(unique_root);

        while !q.is_empty() {
            let id = q.pop_front().unwrap();
            let childs = self.childs(id);
            let mut num_childs = 0;
            for cid in childs {
                let unique_cid = *self.ttree.unique_id(cid).unwrap();
                num_childs += 1;
                q.push_back(unique_cid);
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
    pub fn subtree_leaves(&self, reference: &Reference) -> Vec<TypeTreeNodeIndex> {
        let subtree_root_opt = self.subtree_root(reference);
        if let Some(subtree_root) = subtree_root_opt {
            let graph_root = self.ttree.graph_id(subtree_root).unwrap();
            let subtree = SubTreeView::from_subtree(self, *graph_root);
            subtree.leaves()
        } else {
            vec![]
        }
    }

    pub fn leaves_with_path(&self) -> LeavesWithPath {
        let mut ret = LeavesWithPath::new();
        let mut q: VecDeque<(TypeTreeNodeIndex, TypeTreeNodePath)> = VecDeque::new();

        let unique_root = *self.ttree.unique_id(self.root).unwrap();
        let n = self.root_node().unwrap();
        q.push_back((unique_root, TypeTreeNodePath::new(n.dir, n.tpe.clone(), None)));

        while !q.is_empty() {
            let nirc = q.pop_front().unwrap();

            let mut num_childs = 0;
            for cid in self.childs(nirc.0) {
                let unique_cid = *self.ttree.unique_id(cid).unwrap();
                let cnode = self.get_node(unique_cid).unwrap();
                num_childs += 1;

                let mut path = nirc.1.clone();
                path.append(&cnode);
                q.push_back((unique_cid, path));
            }

            // Leaf node
            if num_childs == 0 {
                ret.insert(nirc.1, nirc.0);
            }
        }
        return ret;
    }

    /// Given a `reference`, returns all the subtree leaf nodes along with the path to each node
    pub fn subtree_leaves_with_path(&self, reference: &Reference) -> LeavesWithPath {
        let subtree_root_opt = self.subtree_root(reference);
        if let Some(subtree_root) = subtree_root_opt {
            let graph_root = self.ttree.graph_id(subtree_root).unwrap();
            let subtree = SubTreeView::from_subtree(self, *graph_root);
            subtree.leaves_with_path()
        } else {
            LeavesWithPath::new()
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
    /// - When `reference` is `io.b`, it will return NodeIndex for `b`
    pub fn subtree_root(&self, reference: &Reference) -> Option<TypeTreeNodeIndex> {
        let mut chain = Self::ref_identifier_chain(reference);
        let mut q: VecDeque<TypeTreeNodeIndex> = VecDeque::new();
        let unique_root = *self.ttree.unique_id(self.root).unwrap();
        q.push_back(unique_root);

        let mut subtree_root_opt: Option<TypeTreeNodeIndex> = None;
        while !q.is_empty() && !chain.is_empty() {
            let id = q.pop_front().unwrap();
            subtree_root_opt = Some(id);

            let node = self.get_node(id).unwrap();
            let ref_name = chain.pop_front().unwrap();
            if id != unique_root && *node.name.as_ref().unwrap() != ref_name {
                self.print_tree();
                panic!("Reference {:?} not inclusive in type tree", reference);
            }

            for cid in self.childs(id) {
                let unique_cid = *self.ttree.unique_id(cid).unwrap();
                let child = self.get_node(unique_cid).unwrap();
                if let Some(ref_name) = chain.front() {
                    if *child.name.as_ref().unwrap() == *ref_name {
                        q.push_back(unique_cid);
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
            let graph_root = *self.ttree.graph_id(subtree_root).unwrap();
            Some(SubTreeView::from_subtree(self, graph_root))
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

    fn eq_recursive(&self, id: NodeIndex, other: &SubTreeView, other_id: NodeIndex) -> bool {
        let unique_id = *self.ttree.unique_id(id).unwrap();
        let node_opt = self.get_node(unique_id);

        let other_unique_id = *other.ttree.unique_id(other_id).unwrap();
        let other_node_opt = other.get_node(other_unique_id);

        match (node_opt, other_node_opt) {
            (Some(node), Some(other_node)) => {
                if node != other_node {
                    false
                } else {
                    let mut childs: Vec<NodeIndex> = self.childs(unique_id).collect();
                    let mut other_childs: Vec<NodeIndex> = other.childs(other_unique_id).collect();
                    if childs.len() != other_childs.len() {
                        false
                    } else {
                        childs.sort_by_key(|&i| {
                            self.get_node(*self.ttree.unique_id(i).unwrap()).unwrap()
                        });
                        other_childs.sort_by_key(|&i| {
                            other.get_node(*other.ttree.unique_id(i).unwrap()).unwrap()
                        });
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

    /// Checks whether all the leaves have the same direction
    pub fn is_bidirectional(&self) -> bool {
        let leaves = self.leaves();
        if leaves.len() <= 1 {
            return false;
        }
        leaves.first()
            .and_then(|first| self.get_node(*first))
            .map_or(false, |first_node| {
                leaves.iter()
                    .map(|id| self.get_node(*id))
                    .any(|node| node.map_or(false, |n| n.dir != first_node.dir))
            })
    }
}
