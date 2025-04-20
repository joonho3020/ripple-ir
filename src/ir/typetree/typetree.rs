use chirrtl_parser::ast::*;
use std::fmt::Debug;
use std::collections::VecDeque;
use petgraph::{graph::{Graph, NodeIndex}, Direction::Outgoing};
use indexmap::IndexMap;
use bimap::BiMap;
use crate::ir::typetree::subtree::SubTreeView;
use crate::ir::typetree::tnode::*;
use crate::ir::typetree::tedge::*;
use crate::ir::IndexGen;

type Tree = Graph<TypeTreeNode, TypeTreeEdge>;

/// A tree that represents the type of an aggregate node
#[derive(Debug, Default, Clone)]
pub struct TypeTree {
    /// Graph representing the tree
    pub graph: Tree,

    /// NodeIndex of the root node
    pub root: Option<NodeIndex>,

    /// Used to generate unique `TypeTreeNodeIndex`
    idx_gen: IndexGen,

    /// Cache that maps petgraph `NodeIndex` to each nodes unique `TypeTreeNodeIndex`
    /// - Needs to be recomputed when nodes are removed
    node_map_cache: BiMap<NodeIndex, TypeTreeNodeIndex>,
}

impl TypeTree {
    /// Build a `TypeTree` that represents a `GroundType`
    pub fn build_from_ground_type(gt: GroundType) -> Self {
        let mut ret = Self::default();
        let node = TypeTreeNode::new(None, TypeDirection::default(), TypeTreeNodeType::Ground(gt));
        let root = ret.graph.add_node(node);
        ret.root = Some(root);
        ret.assign_unique_id();
        return ret;
    }

    /// Build a `TypeTree` that represents a `Type`
    pub fn build_from_type(tpe: &Type, dir: TypeDirection) -> Self {
        let mut ret = Self::default();
        ret.build_recursive(tpe, None, dir, None);
        ret.assign_unique_id();
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
        ret.assign_unique_id();
        return ret;
    }

    /// Assigns a unique id to each node in the `TypeTree`
    pub fn assign_unique_id(&mut self) {
        for id in self.graph.node_indices() {
            let node = self.graph.node_weight_mut(id).unwrap();
            let unique_id = self.idx_gen.generate();
            node.id = Some(unique_id);
            self.node_map_cache.insert(id, unique_id);
        }
    }

    /// Flips the directionality of all the TypeTreeNodes
    pub fn flip(&mut self) {
        for id in self.graph.node_indices() {
            let node = self.graph.node_weight_mut(id).unwrap();
            node.dir = node.dir.flip();
        }
    }

    /// Provides a subtree view of the TypeTree
    pub fn view(&self) -> Option<SubTreeView<'_>> {
        if let Some(root_id) = self.root {
            Some(SubTreeView::new(self, root_id))
        } else {
            None
        }
    }

    pub fn graph_id(&self, id: TypeTreeNodeIndex) -> Option<&NodeIndex> {
        self.node_map_cache.get_by_right(&id)
    }

    pub fn unique_id(&self, id: NodeIndex) -> Option<&TypeTreeNodeIndex> {
        self.node_map_cache.get_by_left(&id)
    }

    /// Reconstruct Type from this typetree
    pub fn to_type(&self) -> Type {
        let root = self.root.expect("TypeTree has no root node");
        self.to_type_recursive(root)
    }

    /// Recursively reconstruct Type from this typetree
    fn to_type_recursive(&self, node_idx: NodeIndex) -> Type {
        let node = &self.graph[node_idx];
        match &node.tpe {
            TypeTreeNodeType::Ground(gt) => {
                Type::TypeGround(gt.clone().into())
            }
            TypeTreeNodeType::Fields => {
                let mut fields = vec![];
                let childs: Vec<NodeIndex> = self.graph.neighbors(node_idx).collect();

                // Iterate in the reverse order to match the original ordering
                for child in childs.iter().rev() {
                    let child_node = &self.graph.node_weight(*child).unwrap();
                    let name = child_node.name.clone().expect("Field must have a name");
                    let child_type = self.to_type_recursive(*child);
                    let field = if child_node.dir == node.dir {
                        Field::Straight(name, Box::new(child_type))
                    } else {
                        Field::Flipped(name, Box::new(child_type))
                    };
                    fields.push(Box::new(field));
                }
                Type::TypeAggregate(Box::new(TypeAggregate::Fields(Box::new(fields))))
            }
            TypeTreeNodeType::Array => {
                let children: Vec<(u32, Type)> = self
                    .graph
                    .neighbors(node_idx)
                    .map(|child| {
                        let child_node = &self.graph.node_weight(child).unwrap();
                        let id = match child_node.name.as_ref().unwrap() {
                            Identifier::ID(x) => x.to_u32(),
                            _ => panic!("Array elements must be named with indices")
                        };
                        let ty = self.to_type_recursive(child);
                        (id, ty)
                    })
                    .collect();

                if children.is_empty() {
                    // There is no children.
                    // Technically, the array can still have a valid type,
                    // but if the array length is zero, the meaning of its
                    // type is unimportant anyways
                    return Type::TypeAggregate(
                        Box::new(TypeAggregate::Fields(Box::new(vec![]))));
                } else {
                    // Assuming homogeneous array
                    let (_, first_type) = children[0].clone();
                    let len = children.len();
                    Type::TypeAggregate(
                        Box::new(TypeAggregate::Array(Box::new(first_type), Int::from(len as u32))))
                }
            }
        }
    }
}