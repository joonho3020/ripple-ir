pub mod typetree;
pub mod whentree;
pub mod firir;

use chirrtl_parser::ast::*;
use derivative::Derivative;
use petgraph::visit::EdgeRef;
use std::fmt::Display;
use std::hash::Hash;
use indexmap::{IndexMap, IndexSet};
use petgraph::graph::{Graph, NodeIndex, EdgeIndex};
use fixedbitset::FixedBitSet;
use crate::common::graphviz::*;
use crate::ir::whentree::Condition;
use crate::ir::typetree::*;
use crate::ir::firir::*;

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct PhiPriority {
    /// Priority between blocks
    /// - Smaller number means higher priority
    pub block: u32,

    /// Priority between statements within the same block
    /// - Smaller number means higher priority
    pub stmt: u32,
}

impl PhiPriority {
    pub fn new(block: u32, stmt: u32) -> Self {
        Self { block, stmt }
    }
}

#[derive(Derivative, Clone)]
#[derivative(Debug)]
pub struct RippleNode {
    pub name: Option<Identifier>,
    pub tpe: RippleNodeType,
    pub tg: GroundType
}

impl RippleNode {
    pub fn new(name: Option<Identifier>, tpe: RippleNodeType, tg: GroundType) -> Self {
        Self { name, tpe, tg }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub enum RippleNodeType {
    #[default]
    Invalid,

    DontCare,

    UIntLiteral(Int),
    SIntLiteral(Int),

    Mux,
    PrimOp2Expr(PrimOp2Expr),
    PrimOp1Expr(PrimOp1Expr),
    PrimOp1Expr1Int(PrimOp1Expr1Int, Int),
    PrimOp1Expr2Int(PrimOp1Expr2Int, Int, Int),

    // Stmt
    Wire,
    Reg,
    RegReset,
    SMem(Option<ChirrtlMemoryReadUnderWrite>),
    CMem,
    WriteMemPort,
    ReadMemPort,
    InferMemPort,
    Inst(Identifier),

    // Port
    Input,
    Output,
    Phi,
}

impl Display for RippleNodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let original = format!("{:?}", self);
        let clean_for_dot = original.replace('"', "");
        write!(f, "{}", clean_for_dot)
    }
}

impl Display for RippleNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let original = format!("{:?}\n{:?}\n{:?}", self.name, self.tpe, self.tg);
        let clean_for_dot = original.replace('"', "");
        write!(f, "{}", clean_for_dot)
    }
}

impl From<&FirNodeType> for RippleNodeType {
    fn from(value: &FirNodeType) -> Self {
        match value {
            FirNodeType::Invalid => Self::Invalid,
            FirNodeType::DontCare => Self::DontCare,
            FirNodeType::UIntLiteral(_, x) => Self::UIntLiteral(x.clone()),
            FirNodeType::SIntLiteral(_, x) => Self::SIntLiteral(x.clone()),
            FirNodeType::Mux => Self::Mux,
            FirNodeType::PrimOp2Expr(op) => Self::PrimOp2Expr(op.clone()),
            FirNodeType::PrimOp1Expr(op) => Self::PrimOp1Expr(op.clone()),
            FirNodeType::PrimOp1Expr1Int(op, a) => Self::PrimOp1Expr1Int(op.clone(), a.clone()),
            FirNodeType::PrimOp1Expr2Int(op, a, b) => Self::PrimOp1Expr2Int(op.clone(), a.clone(), b.clone()),
            FirNodeType::Wire => Self::Wire,
            FirNodeType::Reg => Self::Reg,
            FirNodeType::RegReset => Self::RegReset,
            FirNodeType::SMem(x) => Self::SMem(x.clone()),
            FirNodeType::CMem => Self::CMem,
            FirNodeType::WriteMemPort => Self::WriteMemPort,
            FirNodeType::ReadMemPort => Self::ReadMemPort,
            FirNodeType::InferMemPort => Self::InferMemPort,
            FirNodeType::Inst(x) => Self::Inst(x.clone()),
            FirNodeType::Input => Self::Input,
            FirNodeType::Output => Self::Output,
            FirNodeType::Phi => Self::Phi,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RippleEdge {
    pub width: Option<Width>,
    pub et: RippleEdgeType
}

impl RippleEdge {
    pub fn new(width: Option<Width>, et: RippleEdgeType) -> Self {
        Self { width, et }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RippleEdgeType {
    Wire,

    Operand0,
    Operand1,

    MuxCond,
    MuxTrue,
    MuxFalse,

    Clock,
    Reset,
    DontCare,

    PhiInput(PhiPriority, Condition),
    PhiSel,
    PhiOut,

    MemPortEdge,
    MemPortAddr,

    ArrayAddr,
}

impl Display for RippleEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let original = format!("{:?}", self.et);
        let clean_for_dot = original.replace('"', "");
        write!(f, "{}", clean_for_dot)
    }
}

impl From<&FirEdgeType> for RippleEdgeType {
    fn from(value: &FirEdgeType) -> Self {
        match value {
            FirEdgeType::Wire => RippleEdgeType::Wire,
            FirEdgeType::Operand0 => RippleEdgeType::Operand0,
            FirEdgeType::Operand1 => RippleEdgeType::Operand1,
            FirEdgeType::MuxCond => RippleEdgeType::MuxCond,
            FirEdgeType::MuxTrue => RippleEdgeType::MuxTrue,
            FirEdgeType::MuxFalse => RippleEdgeType::MuxFalse,
            FirEdgeType::Clock => RippleEdgeType::Clock,
            FirEdgeType::Reset => RippleEdgeType::Reset,
            FirEdgeType::DontCare => RippleEdgeType::DontCare,
            FirEdgeType::PhiInput(prior, cond) => RippleEdgeType::PhiInput(prior.clone(), cond.clone()),
            FirEdgeType::PhiSel => RippleEdgeType::PhiSel,
            FirEdgeType::PhiOut => RippleEdgeType::PhiOut,
            FirEdgeType::MemPortEdge => RippleEdgeType::MemPortEdge,
            FirEdgeType::MemPortAddr => RippleEdgeType::MemPortAddr,
            FirEdgeType::ArrayAddr => RippleEdgeType::ArrayAddr
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggEdgeKey {
    pub dst_tree: TreeIdx,
    pub et: RippleEdgeType,
}

impl AggEdgeKey {
    pub fn new(dst_tree: TreeIdx, et: RippleEdgeType) -> Self {
        Self { dst_tree, et }
    }
}

#[derive(Debug, Clone)]
pub struct AggVisMap {
    visited: FixedBitSet
}

impl AggVisMap {
    pub fn new(num_bits: u32) -> Self {
        Self { visited: FixedBitSet::with_capacity(num_bits as usize) }
    }

    pub fn is_visited(&self, id: TreeIdx) -> bool {
        self.visited.contains(id as usize)
    }

    pub fn visit(&mut self, id: TreeIdx) {
        self.visited.set(id as usize, true);
    }
}

type IRGraph = Graph<RippleNode, RippleEdge>;

pub type TreeIdx = u32;

#[derive(Debug, Clone)]
pub struct TypeTreeIdx {
    leaf_id: NodeIndex,
    tree_id: TreeIdx,
}

impl TypeTreeIdx {
    pub fn new(leaf_id: NodeIndex, tree_id: TreeIdx) -> Self {
        Self { leaf_id, tree_id }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RootTypeTreeKey {
    pub name: Identifier,
    pub nt: RippleNodeType,
}

impl RootTypeTreeKey {
    pub fn new(name: Identifier, nt: RippleNodeType) -> Self {
        Self { name, nt }
    }
}

#[derive(Debug, Clone)]
pub struct RippleGraph {
    pub graph: IRGraph,
    pub ttree_idx_map: IndexMap<NodeIndex, TypeTreeIdx>,
    pub root_ref_ttree_idx_map: IndexMap<RootTypeTreeKey, TreeIdx>,
    pub ttree_idx_root_ref_map: IndexMap<TreeIdx, RootTypeTreeKey>,
    pub ttrees: Vec<TypeTree>,
}

impl RippleGraph {
    pub fn new() -> Self {
        Self {
            graph: IRGraph::new(),
            ttree_idx_map: IndexMap::new(),
            root_ref_ttree_idx_map: IndexMap::new(),
            ttree_idx_root_ref_map: IndexMap::new(),
            ttrees: vec![],
        }
    }

    pub fn add_node(&mut self, node: RippleNode) -> NodeIndex {
        self.graph.add_node(node)
    }

    pub fn add_aggregate_node(
        &mut self,
        name: Identifier,
        ttree: &TypeTree,
        nt: RippleNodeType
    ) -> RootTypeTreeKey {
        let mut my_ttree = ttree.clone();
        let ttree_id = self.ttrees.len() as u32;
        let leaves = my_ttree.all_leaves();

        // Add all the leaf nodes
        for leaf_id in leaves {
            let leaf = my_ttree.graph.node_weight(leaf_id).unwrap();
            let tg = match &leaf.tpe {
                TypeTreeNodeType::Ground(x) => x,
                _ => panic!("Type tree leaves doesn't have Ground, got {:?}", leaf)
            };

            let leaf_name = my_ttree.node_name(&name, leaf_id);

            // Insert new graph node
            let rgnode = RippleNode::new(Some(leaf_name), nt.clone(), tg.clone());
            let rg_id = self.add_node(rgnode);

            // Add this node to the ttree_idx_map
            self.ttree_idx_map.insert(rg_id, TypeTreeIdx::new(leaf_id, ttree_id));

            // Update ttree to point to this node
            my_ttree.graph.node_weight_mut(leaf_id).unwrap().id = Some(rg_id);
        }

        println!("-----------------------------");
        println!("{:?} {:?}", name, nt);
        my_ttree.print_tree();

        // Add the type tree
        self.ttrees.push(my_ttree);

        let root_key = RootTypeTreeKey::new(name, nt);
        self.root_ref_ttree_idx_map.insert(root_key.clone(), ttree_id);
        self.ttree_idx_root_ref_map.insert(ttree_id, root_key.clone());
        return root_key;
    }

    pub fn add_edge(&mut self, src: NodeIndex, dst: NodeIndex, edge: RippleEdge) -> EdgeIndex {
        self.graph.add_edge(src, dst, edge)
    }

    pub fn add_single_edge(
        &mut self,
        src_key: &RootTypeTreeKey,
        src_ref: &Reference,
        dst_key: &RootTypeTreeKey,
        dst_ref: &Reference,
        et: RippleEdgeType
    ) {
        println!("src_ref {:?} src_key {:?}", src_ref, src_key);
        println!("dst_ref {:?} dst_key {:?}", dst_ref, dst_key);

        let src_ttree_id = self.root_ref_ttree_idx_map.get(src_key).expect("to exist");
        let src_ttree = self.ttrees.get(*src_ttree_id as usize).expect("to exist");
        let src_leaves = src_ttree.subtree_leaves_with_path(src_ref);

        let dst_ttree_id = self.root_ref_ttree_idx_map.get(dst_key).expect("to exist");
        let dst_ttree = self.ttrees.get(*dst_ttree_id as usize).expect("to exist");
        let dst_leaves = dst_ttree.subtree_leaves_with_path(dst_ref);

        println!("src_ttree");
        src_ttree.print_tree();
        println!("src_leaves {:?}", src_leaves);

        println!("dst_ttree");
        dst_ttree.print_tree();
        println!("dst_leaves {:?}", dst_leaves);

        assert!(src_leaves.len() == 1, "add_single_edge got multiple src_leaves {:?}", src_leaves);
        assert!(dst_leaves.len() > 0, "add_single_edge got zero destinations");

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        let (_src_path_key, src_ttree_leaf_id) = src_leaves.first().unwrap();
        let src_ttree_leaf = src_ttree.graph.node_weight(*src_ttree_leaf_id).unwrap();

        for (_dst_path_key, dst_ttree_leaf_id) in dst_leaves {
            let dst_ttree_leaf = dst_ttree.graph.node_weight(dst_ttree_leaf_id).unwrap();
            if src_ttree_leaf.dir == TypeDirection::Outgoing {
                edges.push((
                        src_ttree_leaf.id.unwrap(),
                        dst_ttree_leaf.id.unwrap(),
                        RippleEdge::new(None, et.clone())));
            } else {
                edges.push((
                        dst_ttree_leaf.id.unwrap(),
                        src_ttree_leaf.id.unwrap(),
                        RippleEdge::new(None, et.clone())));
            }
        }

        for edge in edges {
            self.add_edge(edge.0, edge.1, edge.2);
        }
    }

    pub fn add_aggregate_edge(
        &mut self,
        src_key: &RootTypeTreeKey,
        src_ref: &Reference,
        dst_key: &RootTypeTreeKey,
        dst_ref: &Reference,
        et: RippleEdgeType
    ) {
        println!("src_ref {:?} src_key {:?}", src_ref, src_key);
        println!("dst_ref {:?} dst_key {:?}", dst_ref, dst_key);

        let src_ttree_id = self.root_ref_ttree_idx_map.get(src_key).expect("to exist");
        let src_ttree = self.ttrees.get(*src_ttree_id as usize).expect("to exist");
        let src_leaves = src_ttree.subtree_leaves_with_path(src_ref);

        let dst_ttree_id = self.root_ref_ttree_idx_map.get(dst_key).expect("to exist");
        let dst_ttree = self.ttrees.get(*dst_ttree_id as usize).expect("to exist");
        let dst_leaves = dst_ttree.subtree_leaves_with_path(dst_ref);

        println!("src_leaves: {:?}", src_leaves);
        println!("dst_leaves: {:?}", dst_leaves);

        let mut edges: Vec<(NodeIndex, NodeIndex, RippleEdge)> = vec![];
        for (src_path_key, src_ttree_leaf_id) in src_leaves {
            let src_ttree_leaf = src_ttree.graph.node_weight(src_ttree_leaf_id).unwrap();
            if dst_leaves.contains_key(&src_path_key) {
                let dst_ttree_leaf_id = dst_leaves.get(&src_path_key).unwrap();
                let dst_ttree_leaf = dst_ttree.graph.node_weight(*dst_ttree_leaf_id).unwrap();
                if src_ttree_leaf.dir == TypeDirection::Outgoing {
                    edges.push((
                        src_ttree_leaf.id.unwrap(),
                        dst_ttree_leaf.id.unwrap(),
                        RippleEdge::new(None, et.clone())));
                } else {
                    edges.push((
                        dst_ttree_leaf.id.unwrap(),
                        src_ttree_leaf.id.unwrap(),
                        RippleEdge::new(None, et.clone())));
                }
            } else {
                panic!("Not connected src_ref {:?} src_key {:?} dst_ref {:?} dst_key {:?}",
                    src_key, src_ref, dst_ref, dst_key);
            }
        }

        // TODO: Test for partial connections?

        for edge in edges {
            self.add_edge(edge.0, edge.1, edge.2);
        }
    }

    pub fn check_node_type(&self) {
        for ttree in self.ttrees.iter() {
            let leaf_ids = ttree.all_leaves();
            let mut prev_nt_opt: Option<RippleNodeType> = None;
            for leaf_id in leaf_ids {
                let leaf = ttree.graph.node_weight(leaf_id).unwrap();
                let ir_id = leaf.id.unwrap();
                let ir_node = self.graph.node_weight(ir_id).unwrap();
                if let Some(prev_nt) = prev_nt_opt.clone() {
                    if ir_node.tpe != prev_nt {
                        ttree.print_tree();
                        panic!("Nodes under the same ttree has different types {:?} {:?}", ir_node, prev_nt);
                    }
                } else {
                    prev_nt_opt = Some(ir_node.tpe.clone());
                }
            }
        }
    }

    pub fn node_indices_agg(&self) -> Vec<TreeIdx> {
        (0..self.ttrees.len() as u32).collect()
    }

    pub fn node_weight_agg(&self, id: TreeIdx) -> (Option<&TypeTree>, Option<&RootTypeTreeKey>) {
        (self.ttrees.get(id as usize), self.ttree_idx_root_ref_map.get(&id))
    }

    pub fn neighbors_agg(&self, id: TreeIdx) -> Vec<TreeIdx> {
        let ttree = self.ttrees.get(id as usize).unwrap();
        let leaf_ids = ttree.all_leaves();

        let mut neighbor_ttree_ids: IndexSet<TreeIdx> = IndexSet::new();
        for leaf_id in leaf_ids {
            let leaf = ttree.graph.node_weight(leaf_id).unwrap();
            let ir_id = leaf.id.unwrap();

            let neighbor_ids = self.graph.neighbors(ir_id);
            for nid in neighbor_ids {
                let neighbor_ttree_idx = self.ttree_idx_map.get(&nid).unwrap();
                neighbor_ttree_ids.insert(neighbor_ttree_idx.tree_id);
            }
        }
        return neighbor_ttree_ids.iter().map(|x| *x).collect();
    }

    pub fn neighbors_directed_agg(&self, id: TreeIdx, dir: petgraph::Direction) -> Vec<TreeIdx> {
        let ttree = self.ttrees.get(id as usize).unwrap();
        let leaf_ids = ttree.all_leaves();

        let mut neighbor_ttree_ids: IndexSet<TreeIdx> = IndexSet::new();
        for leaf_id in leaf_ids {
            let leaf = ttree.graph.node_weight(leaf_id).unwrap();
            let ir_id = leaf.id.unwrap();

            let neighbor_ids = self.graph.neighbors_directed(ir_id, dir);
            for nid in neighbor_ids {
                let neighbor_ttree_idx = self.ttree_idx_map.get(&nid).unwrap();
                neighbor_ttree_ids.insert(neighbor_ttree_idx.tree_id);
            }
        }
        return neighbor_ttree_ids.iter().map(|x| *x).collect();
    }

    pub fn edges_directed_agg(&self, id: TreeIdx, dir: petgraph::Direction) -> IndexMap<AggEdgeKey, Vec<EdgeIndex>> {
        let ttree = self.ttrees.get(id as usize).unwrap();
        let leaf_ids = ttree.all_leaves();

        let mut ttree_edge_map: IndexMap<AggEdgeKey, Vec<EdgeIndex>> = IndexMap::new();
        for leaf_id in leaf_ids {
            let leaf = ttree.graph.node_weight(leaf_id).unwrap();
            let ir_id = leaf.id.unwrap();

            let edge_ids = self.graph.edges_directed(ir_id, dir);
            for eid in edge_ids {
                let (_src, dst) = self.graph.edge_endpoints(eid.id()).unwrap();
                let neighbor_ttree_idx = self.ttree_idx_map.get(&dst).unwrap();
                let rir_edge = self.graph.edge_weight(eid.id()).unwrap();

                let edge_map_key = AggEdgeKey::new(neighbor_ttree_idx.tree_id, rir_edge.et.clone());
                if !ttree_edge_map.contains_key(&edge_map_key) {
                    ttree_edge_map.insert(edge_map_key.clone(), vec![]);
                }
                ttree_edge_map
                    .get_mut(&edge_map_key)
                    .unwrap()
                    .push(eid.id());
            }
        }
        return ttree_edge_map;
    }

    pub fn edges_agg(&self, id: TreeIdx) -> IndexMap<AggEdgeKey, Vec<EdgeIndex>> {
        let ttree = self.ttrees.get(id as usize).unwrap();
        let leaf_ids = ttree.all_leaves();

        let mut ttree_edge_map: IndexMap<AggEdgeKey, Vec<EdgeIndex>> = IndexMap::new();
        for leaf_id in leaf_ids {
            let leaf = ttree.graph.node_weight(leaf_id).unwrap();
            let ir_id = leaf.id.unwrap();

            let edge_ids = self.graph.edges(ir_id);
            for eid in edge_ids {
                let (_src, dst) = self.graph.edge_endpoints(eid.id()).unwrap();
                let neighbor_ttree_idx = self.ttree_idx_map.get(&dst).unwrap();
                let rir_edge = self.graph.edge_weight(eid.id()).unwrap();

                let edge_map_key = AggEdgeKey::new(neighbor_ttree_idx.tree_id, rir_edge.et.clone());
                if !ttree_edge_map.contains_key(&edge_map_key) {
                    ttree_edge_map.insert(edge_map_key.clone(), vec![]);
                }
                ttree_edge_map
                    .get_mut(&edge_map_key)
                    .unwrap()
                    .push(eid.id());
            }
        }
        return ttree_edge_map;
    }

    pub fn vismap_agg(&self) -> AggVisMap {
        AggVisMap::new(self.ttrees.len() as u32)
    }
}

#[derive(Debug, Clone)]
pub struct RippleIR {
    pub name: Identifier,
    pub graphs: IndexMap<Identifier, RippleGraph>
}

impl RippleIR {
    pub fn new(name: Identifier) -> Self {
        Self { name, graphs: IndexMap::new() }
    }
}

impl GraphViz for RippleGraph {
    fn graphviz_string(
        self: &Self,
        node_attr: Option<&NodeAttributeMap>,
        edge_attr: Option<&EdgeAttributeMap>
    ) -> Result<String, std::io::Error> {
        use graphviz_rust::{
            attributes::{rankdir, EdgeAttributes, GraphAttributes, NodeAttributes},
            dot_generator::{edge, id, node_id},
            dot_structures::Id,
            dot_structures::Stmt as DotStmt,
            dot_structures::Subgraph as DotSubgraph,
            dot_structures::Node as DotNode,
            dot_structures::NodeId as DotNodeId,
            dot_structures::*,
            printer::{DotPrinter, PrinterContext}
        };

        let mut g = graphviz_rust::dot_structures::Graph::DiGraph {
            id: graphviz_rust::dot_generator::id!(""),
            strict: false,
            stmts: vec![
                DotStmt::from(GraphAttributes::rankdir(rankdir::TB)),
                DotStmt::from(GraphAttributes::splines(true)),
                DotStmt::from(GraphAttributes::mindist(2.0)),
                DotStmt::from(GraphAttributes::ranksep(5.0)),
            ]
        };

        // Add nodes
        for (ttree_idx, ttree) in self.ttrees.iter().enumerate() {
            let leaves = ttree.all_leaves();
            let root_info = self.ttree_idx_root_ref_map.get(&(ttree_idx as TreeIdx)).unwrap();

            // Create graphviz subgraph to group nodes together
            let subgraph_name = format!("\"cluster_{}\"", root_info.name).replace('"', "");
            let mut subgraph = DotSubgraph {
                id: Id::Plain(subgraph_name),
                stmts: vec![]
            };

            // Collect all flattened nodes under the current aggregate node
            for ttree_id in leaves.iter() {
                let leaf = ttree.graph.node_weight(*ttree_id).unwrap();
                let rir_id = leaf.id.unwrap();
                let rir_node = self.graph.node_weight(rir_id).unwrap();

                let node_label_inner = format!("{}", rir_node).to_string().replace('"', "");
                let node_label = format!("\"{}\"", node_label_inner);


                // Create graphviz node
                let mut gv_node = DotNode {
                    id: DotNodeId(Id::Plain(rir_id.index().to_string()), None),
                    attributes: vec![
                        NodeAttributes::label(node_label)
                    ],
                };

                // Add node attribute if it exists
                if let Some(na) = node_attr {
                    if na.contains_key(&rir_id) {
                        gv_node.attributes.push(na.get(&rir_id).unwrap().clone());
                    }
                }
                subgraph.stmts.push(DotStmt::from(gv_node));
            }
            g.add_stmt(DotStmt::from(DotSubgraph::from(subgraph)));
        }

        // Add edges
        for eid in self.graph.edge_indices() {
            let ep = self.graph.edge_endpoints(eid).unwrap();
            let w = self.graph.edge_weight(eid).unwrap();
            // Create graphviz edge
            let mut e = edge!(
                node_id!(ep.0.index().to_string()) =>
                node_id!(ep.1.index().to_string()));

            let edge_label_inner = format!("{}", w).to_string().replace('"', "");
            let edge_label = format!("\"{}\"", edge_label_inner);

            e.attributes.push(
                EdgeAttributes::label(edge_label));

            // Add edge attribute if it exists
            if let Some(ea) = edge_attr {
                if ea.contains_key(&eid) {
                    e.attributes.push(ea.get(&eid).unwrap().clone());
                }
            }

            g.add_stmt(Stmt::Edge(e));
        }

        // Export to pdf
        let dot = g.print(&mut PrinterContext::new(true, 4, "\n".to_string(), 90));
        Ok(dot)
    }
}
