use petgraph::graph::EdgeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction::Incoming;
use petgraph::Direction::Outgoing;
use chirrtl_parser::ast::*;
use derivative::Derivative;
use indexmap::IndexMap;
use petgraph::graph::{Graph, NodeIndex};
use crate::common::graphviz::*;
use crate::common::RippleIRErr;
use crate::ir::typetree::typetree::*;
use crate::impl_clean_display;

use super::whentree::PrioritizedCond;

#[derive(Derivative, Clone)]
#[derivative(Debug)]
pub struct FirNode {
    pub name: Option<Identifier>,

    pub nt: FirNodeType,

    #[derivative(Debug="ignore")]
    pub ttree: Option<TypeTree>,
}

impl FirNode {
    pub fn new(name: Option<Identifier>, nt: FirNodeType, ttree: Option<TypeTree>) -> Self {
        Self { name, nt, ttree }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Hash)]
pub enum FirNodeType {
    #[default]
    Invalid,

    DontCare,

    UIntLiteral(Width, Int),
    SIntLiteral(Width, Int),

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
    WriteMemPort(PrioritizedCond),
    ReadMemPort(PrioritizedCond),
    InferMemPort(PrioritizedCond),
    Inst(Identifier),

    // Port
    Input,
    Output,
    Phi,
}

impl_clean_display!(FirNode);

#[derive(Debug, Clone, PartialEq)]
pub struct FirEdge {
    pub src: Expr,
    pub dst: Option<Reference>,
    pub et: FirEdgeType
}

#[derive(Derivative, Clone, PartialEq, Hash)]
#[derivative(Debug)]
pub enum FirEdgeType {
    Wire,

    Operand0,
    Operand1,

    MuxCond,
    MuxTrue,
    MuxFalse,

    Clock,
    Reset,
    DontCare,

    PhiInput(PrioritizedCond),
    PhiSel,
    PhiOut,

    MemPortEdge,
    MemPortAddr,
    MemPortEn,

    ArrayAddr,
}

impl_clean_display!(FirEdge);

impl FirEdge {
    pub fn new(src: Expr, dst: Option<Reference>, et: FirEdgeType) -> Self {
        Self { src, et, dst }
    }
}

type IRGraph = Graph<FirNode, FirEdge>;

#[derive(Debug, Clone)]
pub struct ExtModuleInfo {
    pub defname: DefName,
    pub params: Parameters,
}

impl From<&ExtModule> for ExtModuleInfo {
    fn from(value: &ExtModule) -> Self {
        ExtModuleInfo {
            defname: value.defname.clone(),
            params: value.params.clone()
        }
    }
}

#[derive(Debug, Clone)]
pub struct FirGraph {
    pub graph: IRGraph,
    pub ext_info: Option<ExtModuleInfo>,
    pub blackbox: bool,
}

impl FirGraph {
    pub fn new(blackbox: bool) -> Self {
        Self { graph: IRGraph::new(), blackbox, ext_info: None }
    }

    /// Returns a TypeTree of all its IO signals
    /// Useful for handling instances
    pub fn io_typetree(&self) -> TypeTree {
        let mut io_ttrees: IndexMap<&Identifier, &TypeTree> = IndexMap::new();
        for id in self.graph.node_indices() {
            let node = self.graph.node_weight(id).unwrap();
            if node.nt == FirNodeType::Input || node.nt == FirNodeType::Output {
                io_ttrees.insert(&node.name.as_ref().unwrap(), &node.ttree.as_ref().unwrap());
            }
        }
        TypeTree::merge_trees(io_ttrees)
    }

    /// Returns an `EdgeIndex` comming into the node that matches `et` if exists
    pub fn parent_with_type(&self, id: NodeIndex, et: FirEdgeType) -> Option<EdgeIndex> {
        for eid in self.graph.edges_directed(id, Incoming) {
            let edge = self.graph.edge_weight(eid.id()).unwrap();
            if edge.et == et {
                return Some(eid.id());
            }
        }
        return None;
    }

    /// Returns an `EdgeIndex` going out of node that matches `et` if exists
    pub fn childs_with_type(&self, id: NodeIndex, et: FirEdgeType) -> Vec<EdgeIndex> {
        let mut ret: Vec<EdgeIndex> = vec![];
        for eid in self.graph.edges_directed(id, Outgoing) {
            let edge = self.graph.edge_weight(eid.id()).unwrap();
            if edge.et == et {
                ret.push(eid.id());
            }
        }
        return ret;
    }
}

#[derive(Debug, Clone)]
pub struct FirIR {
    pub version: Version,
    pub name: Identifier,
    pub annos: Annotations,
    pub graphs: IndexMap<Identifier, FirGraph>
}

impl FirIR {
    pub fn new(version: Version, name: Identifier, annos: Annotations) -> Self {
        Self { version, name, annos, graphs: IndexMap::new() }
    }

    pub fn export(&self, outdir: &str, pfx: &str) -> Result<(), RippleIRErr> {
        for (module, fg) in self.graphs.iter() {
            fg.export_graphviz(
                &format!("{}/{}-{}.{}.pdf", outdir, self.name.to_string(), module, pfx),
                None,
                None,
                false)?;
        }
        Ok(())
    }
}

impl DefaultGraphVizCore<FirNode, FirEdge> for FirGraph {
    fn node_indices(self: &Self) -> petgraph::graph::NodeIndices {
        self.graph.node_indices()
    }

    fn node_weight(self: &Self, id: NodeIndex) -> Option<&FirNode> {
        self.graph.node_weight(id)
    }

    fn edge_indices(self: &Self) -> petgraph::graph::EdgeIndices {
        self.graph.edge_indices()
    }

    fn edge_endpoints(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.graph.edge_endpoints(id)
    }

    fn edge_weight(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<&FirEdge> {
        self.graph.edge_weight(id)
    }
}

impl GraphViz for FirGraph {
    fn graphviz_string(
            self: &Self,
            node_attr: Option<&NodeAttributeMap>,
            edge_attr: Option<&EdgeAttributeMap>
    ) -> Result<String, std::io::Error> {
        DefaultGraphVizCore::graphviz_string(self, node_attr, edge_attr)
    }
}

#[cfg(test)]
mod test {
    use crate::common::RippleIRErr;
    use crate::ir::typetree::typetree::*;
    use crate::ir::typetree::tnode::*;
    use crate::ir::typetree::tedge::*;
    use chirrtl_parser::ast::*;
    use chirrtl_parser::parse_circuit;
    use crate::passes::fir::from_ast::from_circuit;
    use crate::passes::fir::remove_unnecessary_phi::remove_unnecessary_phi;
    use crate::passes::fir::check_phi_nodes::check_phi_node_connections;

    #[test]
    fn io_typetree() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/GCD.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let mut fir = from_circuit(&circuit);
        remove_unnecessary_phi(&mut fir);
        check_phi_node_connections(&fir)?;

        for (_name, fg) in fir.graphs {
            let mut io_typetree = fg.io_typetree();
            io_typetree.flip();

            let mut expect = TypeTree::default();
            let root_id = expect.graph.add_node(TypeTreeNode::new(
                    None, TypeDirection::Incoming, TypeTreeNodeType::Fields));
            expect.root = Some(root_id);

            let io_id = expect.graph.add_node(TypeTreeNode::new(
                    Some(Identifier::Name("io".to_string())),
                    TypeDirection::Outgoing,
                    TypeTreeNodeType::Fields));

            expect.graph.add_edge(root_id, io_id, TypeTreeEdge::default());

            let value1_id = expect.graph.add_node(TypeTreeNode::new(
                    Some(Identifier::Name("value1".to_string())),
                    TypeDirection::Incoming,
                    TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(16))))));

            let value2_id = expect.graph.add_node(TypeTreeNode::new(
                    Some(Identifier::Name("value2".to_string())),
                    TypeDirection::Incoming,
                    TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(16))))));

            let loadingvalues_id = expect.graph.add_node(TypeTreeNode::new(
                    Some(Identifier::Name("loadingValues".to_string())),
                    TypeDirection::Incoming,
                    TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(1))))));

            let outputgcd_id = expect.graph.add_node(TypeTreeNode::new(
                    Some(Identifier::Name("outputGCD".to_string())),
                    TypeDirection::Outgoing,
                    TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(16))))));

            let outputvalid_id = expect.graph.add_node(TypeTreeNode::new(
                    Some(Identifier::Name("outputValid".to_string())),
                    TypeDirection::Outgoing,
                    TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(1))))));

            expect.graph.add_edge(io_id, value1_id, TypeTreeEdge::default());
            expect.graph.add_edge(io_id, value2_id, TypeTreeEdge::default());
            expect.graph.add_edge(io_id, loadingvalues_id, TypeTreeEdge::default());
            expect.graph.add_edge(io_id, outputgcd_id, TypeTreeEdge::default());
            expect.graph.add_edge(io_id, outputvalid_id, TypeTreeEdge::default());

            let clock_id = expect.graph.add_node(
                TypeTreeNode::new(Some(Identifier::Name("clock".to_string())),
                TypeDirection::Incoming,
                TypeTreeNodeType::Ground(GroundType::Clock)));

            let reset_id = expect.graph.add_node(
                TypeTreeNode::new(Some(Identifier::Name("reset".to_string())),
                TypeDirection::Incoming,
                TypeTreeNodeType::Ground(GroundType::UInt(Some(Width(1))))));

            expect.graph.add_edge(root_id, clock_id, TypeTreeEdge::default());
            expect.graph.add_edge(root_id, reset_id, TypeTreeEdge::default());
            expect.assign_unique_id();

            assert!(io_typetree.view().unwrap().eq(&expect.view().unwrap()));
        }

        Ok(())
    }
}
