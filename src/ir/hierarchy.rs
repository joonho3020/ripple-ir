use chirrtl_parser::ast::Identifier;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::algo::toposort;
use indexmap::IndexMap;

use super::fir::{FirIR, FirNodeType};
use crate::common::graphviz::*;
use crate::impl_clean_display;

/// Represents a node in the `Hierarchy`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HierNode(Identifier);

impl From<Identifier> for HierNode {
    fn from(value: Identifier) -> Self {
        Self(value)
    }
}

impl_clean_display!(HierNode);

impl HierNode {
    pub fn name(&self) -> &Identifier {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HierEdge(Identifier);

impl From<Identifier> for HierEdge {
    fn from(value: Identifier) -> Self {
        Self(value)
    }
}

impl_clean_display!(HierEdge);

type DAG = Graph<HierNode, HierEdge>;

/// A tree that represents the module hierarchy
#[derive(Debug, Default, Clone)]
pub struct Hierarchy {
    /// Graph representing the hierarchy
    /// - Nodes: modules
    /// - Edges: instance name of the child node
    pub graph: DAG,

    /// NodeIndex of the root node
    pub root: Option<NodeIndex>
}

impl Hierarchy {
    pub fn build_from_fir(&mut self, fir: &FirIR) {
        let mut module_node_map: IndexMap<&Identifier, NodeIndex> = IndexMap::new();

        for (module, fg) in fir.graphs.iter() {
            let hnode = HierNode(module.clone());
            let nid = self.graph.add_node(hnode);
            module_node_map.insert(module, nid);

            for id in fg.graph.node_indices() {
                let node = fg.graph.node_weight(id).unwrap();
                match &node.nt {
                    FirNodeType::Inst(child_module) => {
                        self.graph.add_edge(nid,
                            *module_node_map.get(child_module).unwrap(),
                            HierEdge::from(node.name.as_ref().unwrap().clone()));
                    }
                    _ => {
                    }
                }
            }
        }
    }

    pub fn new(fir: &FirIR) -> Self {
        let mut ret = Self::default();
        ret.build_from_fir(fir);
        return ret;
    }

    /// Returns a iterator over the HierNodes in the module hierarchy
    /// in a topological order (from leaf to top)
    pub fn topo_order(&self) -> impl Iterator<Item = &HierNode> {
        let sorted = toposort(&self.graph, None).expect("Hier graph is a DAG");
        sorted.into_iter().rev().map(|id| self.graph.node_weight(id).unwrap())
    }
}

impl DefaultGraphVizCore<HierNode, HierEdge> for Hierarchy {
    fn node_indices(self: &Self) -> petgraph::graph::NodeIndices {
        self.graph.node_indices()
    }

    fn node_weight(self: &Self, id: NodeIndex) -> Option<&HierNode> {
        self.graph.node_weight(id)
    }

    fn edge_indices(self: &Self) -> petgraph::graph::EdgeIndices {
        self.graph.edge_indices()
    }

    fn edge_endpoints(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        self.graph.edge_endpoints(id)
    }

    fn edge_weight(self: &Self, id: petgraph::prelude::EdgeIndex) -> Option<&HierEdge> {
        self.graph.edge_weight(id)
    }
}

impl GraphViz for Hierarchy {
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
    use crate::passes::fir::from_ast::from_circuit;
    use crate::passes::fir::remove_unnecessary_phi::*;
    use crate::passes::fir::check_phi_nodes::*;

    use super::*;
    use chirrtl_parser::parse_circuit;

    #[test]
    fn build() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/Hierarchy.fir".to_string())?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let mut fir = from_circuit(&circuit);
        remove_unnecessary_phi(&mut fir);
        check_phi_node_connections(&fir)?;

        let h = Hierarchy::new(&fir);
        let hns: Vec<&HierNode> = h.topo_order().collect();

        assert_eq!(hns,
            vec![&HierNode::from(Identifier::Name("A".to_string())),
                 &HierNode::from(Identifier::Name("C".to_string())),
                 &HierNode::from(Identifier::Name("B".to_string())),
                 &HierNode::from(Identifier::Name("Top".to_string()))]);
        Ok(())
    }
}
