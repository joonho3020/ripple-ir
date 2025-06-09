use rusty_firrtl::Identifier;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::algo::toposort;
use indexmap::IndexMap;
use petgraph::Direction::Incoming;
use petgraph::visit::Bfs;
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

    pub fn set_name(&mut self, name: Identifier) {
        self.0 = name;
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
}

impl Hierarchy {
    pub fn build_from_fir(&mut self, fir: &FirIR) {
        let mut module_node_map: IndexMap<&Identifier, NodeIndex> = IndexMap::new();

        for (module, _fg) in fir.graphs.iter() {
            let hnode = HierNode(module.clone());
            let nid = self.graph.add_node(hnode);
            module_node_map.insert(module, nid);
        }

        for (module, fg) in fir.graphs.iter() {
            let nid = *module_node_map.get(module).unwrap();
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

    /// Returns the id of the top module
    pub fn top(&self) -> Option<NodeIndex> {
        let mut ret = None;
        for id in self.graph.node_indices() {
            let incoming = self.graph.neighbors_directed(id, Incoming);
            if incoming.count() == 0 {
                assert!(ret == None, "Should only have one node in the hierarchy with no parent");
                ret = Some(id);
            }
        }
        return ret;
    }

    /// Returns the name of the top module
    pub fn top_name(&self) -> &Identifier {
        self.graph.node_weight(self.top().unwrap()).unwrap().name()
    }

    /// Constructs a new hierarchy from the FIR representation
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

    /// Returns the id of the node with matching module name if it exists
    pub fn id(&self, module: &Identifier) -> Option<NodeIndex> {
        for id in self.graph.node_indices() {
            let node = self.graph.node_weight(id).unwrap();
            if node.name() == module {
                return Some(id);
            }
        }
        None
    }

    /// Returns all ids of the node under (and including) module
    pub fn all_childs(&self, module: &Identifier) -> Vec<NodeIndex> {
        let mut ret = vec![];
        if let Some(id) = self.id(module) {
            ret.push(id);

            let mut bfs = Bfs::new(&self.graph, id);
            while let Some(nx) = bfs.next(&self.graph) {
                ret.push(nx);
            }
        }
        ret
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

#[derive(Debug)]
pub struct InstPath {
    pub module: String,
    pub inst: Option<String>,
}

impl InstPath {
    pub fn parse_inst_hierarchy_path(path: &str) -> Vec<InstPath> {
        path.split('/')
            .map(|segment| {
                let parts: Vec<&str> = segment.split(':').collect();
                match parts.len() {
                    2 => InstPath {
                        inst: Some(parts[0].to_string()),
                        module: parts[1].to_string(),
                    },
                    1 => InstPath {
                        inst: None,
                        module: parts[0].to_string(),
                    },
                    _ => panic!("Invalid segment format: {}", segment),
                }
            })
        .collect()
    }
}

#[cfg(test)]
mod test {
    use crate::common::RippleIRErr;
    use crate::passes::runner::run_fir_passes_from_circuit;

    use super::*;
    use chirrtl_parser::parse_circuit;

    #[test]
    fn build() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/Hierarchy.fir".to_string())?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let fir = run_fir_passes_from_circuit(&circuit)?;
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
