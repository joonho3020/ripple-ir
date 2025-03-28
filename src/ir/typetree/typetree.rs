use chirrtl_parser::ast::*;
use std::fmt::Debug;
use std::collections::VecDeque;
use petgraph::{graph::{Graph, NodeIndex}, Direction::Outgoing};
use indexmap::IndexMap;
use crate::common::graphviz::*;
use crate::ir::typetree::subtree::SubTreeView;
use crate::ir::typetree::tnode::*;
use crate::ir::typetree::tedge::*;

type Tree = Graph<TypeTreeNode, TypeTreeEdge>;

pub type TTreeNodeIndex = NodeIndex;

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
        parent_opt: Option<TTreeNodeIndex>,
    ) -> TTreeNodeIndex {
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
        parent_opt: Option<TTreeNodeIndex>
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
            let mut node_id_map: IndexMap<TTreeNodeIndex, TTreeNodeIndex> = IndexMap::new();

            // Add name to the subtree root
            let root_id = ttree.root.unwrap();
            let mut root = ttree.graph.node_weight(root_id).unwrap().clone();
            root.name = Some(name.clone());

            // Add the subtree root as a child to the root
            let new_id = ret.graph.add_node(root);
            ret.graph.add_edge(ret.root.unwrap(), new_id, TypeTreeEdge::default());
            node_id_map.insert(root_id, new_id);

            // Traverse the subtree and add all its childs
            let mut q: VecDeque<TTreeNodeIndex> = VecDeque::new();
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

    pub fn view(&self) -> Option<SubTreeView<'_>> {
        if let Some(root_id) = self.root {
            Some(SubTreeView::new(self, root_id))
        } else {
            None
        }
    }
}

impl DefaultGraphVizCore<TypeTreeNode, TypeTreeEdge> for TypeTree {
    fn node_indices(self: &Self) -> petgraph::graph::NodeIndices {
        self.graph.node_indices()
    }

    fn node_weight(self: &Self, id: TTreeNodeIndex) -> Option<&TypeTreeNode> {
        self.graph.node_weight(id)
    }

    fn edge_indices(self: &Self) -> petgraph::graph::EdgeIndices {
        self.graph.edge_indices()
    }

    fn edge_endpoints(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<(TTreeNodeIndex, TTreeNodeIndex)> {
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
