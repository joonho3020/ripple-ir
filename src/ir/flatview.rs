use crate::ir::*;
use crate::ir::typetree::*;
use indexmap::IndexMap;
use indextree::NodeId;
use petgraph::graph::{EdgeIndex, Graph, NodeIndex};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FlatViewNode {
    pub tid: Option<NodeId>,
    pub nid: NodeIndex,
}


#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FlatViewEdge {
    pub tid: Option<NodeId>,
    pub eid: EdgeIndex,
}

#[derive(Debug, Default)]
pub struct FlatViewGraphBuilder {
    pub graph: FlatViewGraph,
    pub hier2flat: IndexMap<FlatViewNode, NodeIndex>,
    pub nodetree: IndexMap<NodeIndex, TypeTree>,
    pub edgetree: IndexMap<EdgeIndex, TypeTree>,
}

type FlatViewGraph = Graph<FlatViewNode, FlatViewEdge>;

impl RippleGraph {
    fn add_flatview_nodes(&self, builder: &mut FlatViewGraphBuilder) {
        for id in self.graph.node_indices() {
            let node = self.graph.node_weight(id).unwrap();
            match node {
                NodeType::Invalid                 |
                    NodeType::DontCare            |
                    NodeType::UIntLiteral(..)     |
                    NodeType::SIntLiteral(..)     |
                    NodeType::Mux                 |
                    NodeType::PrimOp2Expr(..)     |
                    NodeType::PrimOp1Expr(..)     |
                    NodeType::PrimOp1Expr1Int(..) |
                    NodeType::PrimOp1Expr2Int(..) |
                    NodeType::WriteMemPort(..)    |
                    NodeType::ReadMemPort(..)     |
                    NodeType::InferMemPort(..)    |
                    NodeType::Inst(..)            |
                    NodeType::SMem(..)            |
                    NodeType::CMem(..)            |
                    NodeType::Phi(..) => {
                    let fv = FlatViewNode { tid: None, nid: id };
                    let fid = builder.graph.add_node(fv.clone());
                    builder.hier2flat.insert(fv, fid);
                }
                NodeType::Input(name, tpe)       |
                    NodeType::Output(name, tpe)  |
                    NodeType::Wire(name, tpe)    |
                    NodeType::Reg(name, tpe, ..) |
                    NodeType::RegReset(name, tpe, ..) => {
                    let tree = TypeTree::construct_tree(tpe, name.clone(), Direction::from(node));
                    let leaf_nodes = tree.collect_leaf_nodes();
                    for leaf in leaf_nodes {
                        let fv = FlatViewNode { tid: Some(leaf), nid: id };
                        let fid = builder.graph.add_node(fv.clone());
                        builder.hier2flat.insert(fv, fid);
                    }
                    builder.nodetree.insert(id, tree);
                }
            }
        }
    }

    fn add_flatview_edges(&self, builder: &mut FlatViewGraphBuilder) {
        for id in self.graph.edge_indices() {
            let edge = self.graph.edge_weight(id).unwrap();
            match edge {
                EdgeType::Operand0(expr)     |
                    EdgeType::Operand1(expr) |
                    EdgeType::MuxCond(expr) => {
                    let ep = self.graph.edge_endpoints(id).unwrap();
                    let src_fv = if let Some(src_nodetree) = builder.nodetree.get(&ep.0) {
                        FlatViewNode { tid: None, nid: ep.0 };
                    } else {
                        FlatViewNode { tid: None, nid: ep.0 };
                    };
                    let dst_fv = FlatViewNode { tid: None, nid: ep.1 };
// let src_fid = builder.hier2flat.get(&src_fv).unwrap();
                    let dst_fid = builder.hier2flat.get(&dst_fv).unwrap();
// builder.graph.add_edge(src_fid, dst_fid, FlatViewEdge {
                }
                EdgeType::Clock(..)       |
                    EdgeType::Reset(..)       |
                    EdgeType::DontCare(..)    |
                    EdgeType::PhiSel(..)      |
                    EdgeType::MemPortEdge     |
                    EdgeType::MemPortAddr(..) |
                    EdgeType::ArrayAddr(..) => {
                }
                EdgeType::Wire(r, e) => {
                    self.add_flatview_input_edges_from_expr(id, e, builder);
                    self.add_flatview_output_edges_from_ref(id, r, builder);
                }
                EdgeType::PhiOut(r) => {
                    self.add_flatview_output_edges_from_ref(id, r, builder);
                }
                EdgeType::PhiInput(_pri, _cond, _r, e) => {
                    self.add_flatview_input_edges_from_expr(id, e, builder);
                }
                EdgeType::MuxTrue(e) |
                    EdgeType::MuxFalse(e) => {
                    self.add_flatview_input_edges_from_expr(id, e, builder);
                }
            }
        }
    }

    fn add_flatview_input_edges_from_expr(
        &self,
        id: EdgeIndex,
        e: &Expr,
        builder: &mut FlatViewGraphBuilder
    ) {
        let ep = self.graph.edge_endpoints(id).unwrap();
    }

    fn add_flatview_output_edges_from_ref(
        &self,
        id: EdgeIndex,
        r: &Reference,
        builder: &mut FlatViewGraphBuilder
    ) {
    }

    pub fn get_flat_view(&self) -> FlatViewGraphBuilder {
        let mut builder = FlatViewGraphBuilder::default();
        self.add_flatview_nodes(&mut builder);
        self.add_flatview_edges(&mut builder);
        return builder;
    }
}
