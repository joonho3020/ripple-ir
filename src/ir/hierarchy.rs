use chirrtl_parser::ast::Identifier;
use petgraph::graph::{Graph, NodeIndex};
use indexmap::IndexMap;

use super::firir::{FirIR, FirNodeType};
use crate::common::graphviz::*;
use crate::impl_clean_display;

#[derive(Debug, Clone)]
pub struct HierNode(Identifier);

impl From<Identifier> for HierNode {
    fn from(value: Identifier) -> Self {
        Self(value)
    }
}

impl_clean_display!(HierNode);

#[derive(Debug, Clone)]
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
    pub graph: DAG,

    /// NodeIndex of the root node
    pub root: Option<NodeIndex>
}

impl Hierarchy {
    pub fn build_from_fir(fir: &FirIR) -> Self {
        let mut ret = Self::default();
        let mut module_node_map: IndexMap<&Identifier, NodeIndex> = IndexMap::new();

        for (module, fg) in fir.graphs.iter() {
            let hnode = HierNode(module.clone());
            let nid = ret.graph.add_node(hnode);
            module_node_map.insert(module, nid);

            for id in fg.graph.node_indices() {
                let node = fg.graph.node_weight(id).unwrap();
                match &node.nt {
                    FirNodeType::Inst(child_module) => {
                        ret.graph.add_edge(nid,
                            *module_node_map.get(child_module).unwrap(),
                            HierEdge::from(node.name.as_ref().unwrap().clone()));
                    }
                    _ => {
                    }
                }
            }
        }
        return ret;
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
    use crate::passes::from_ast::from_circuit;
    use crate::passes::remove_unnecessary_phi::*;
    use crate::passes::check_phi_nodes::*;

    use super::Hierarchy;
    use chirrtl_parser::parse_circuit;

    #[test]
    fn build() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/Hierarchy.fir".to_string())?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let mut fir = from_circuit(&circuit);

        remove_unnecessary_phi(&mut fir);
        check_phi_node_connections(&fir)?;

        let _h = Hierarchy::build_from_fir(&fir);
        Ok(())
    }
}
