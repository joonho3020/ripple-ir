use std::fmt::Display;
use graphviz_rust::{
    attributes::{rankdir, EdgeAttributes, GraphAttributes, NodeAttributes},
    cmd::{CommandArg, Format},
    dot_generator::{edge, id, node_id},
    dot_structures::*,
    exec_dot,
    printer::{DotPrinter, PrinterContext}
};
use indexmap::IndexMap;
use petgraph::graph::{EdgeIndex, EdgeIndices, NodeIndex, NodeIndices};

/// IndexMap from Lgraph node index to a GraphViz node attribute.
/// The attributes are added when passing this value to `export_graphviz`
pub type NodeAttributeMap = IndexMap<NodeIndex, Attribute>;

pub trait GraphViz<N, E>
where
    N: Display,
    E: Display,
{
    fn node_indices(self: &Self) -> NodeIndices;
    fn node_weight(self: &Self, id: NodeIndex) -> Option<&N>;
    fn edge_indices(self: &Self) -> EdgeIndices;
    fn edge_endpoints(self: &Self, id: EdgeIndex) -> Option<(NodeIndex, NodeIndex)>;
    fn edge_weight(self: &Self, id: EdgeIndex) -> Option<&E>;

    /// Exports the graph into a graphviz pdf output
    /// - path: should include the filename
    /// - node_attr: adds additional node attributes when provided
    fn export_graphviz(
        self: &Self,
        path: &str,
        node_attr: Option<&NodeAttributeMap>,
        debug: bool
    ) -> Result<String, std::io::Error> {
        let dot = self.graphviz_string(node_attr)?;
        if debug {
            println!("{}", dot);
        }
        exec_dot(dot.clone(), vec![Format::Pdf.into(), CommandArg::Output(path.to_string())])?;
        return Ok(dot);
    }

    fn graphviz_string(
        self: &Self,
        node_attr: Option<&NodeAttributeMap>
    ) -> Result<String, std::io::Error> {
        let mut g = graphviz_rust::dot_structures::Graph::DiGraph {
            id: graphviz_rust::dot_generator::id!(""),
            strict: false,
            stmts: vec![
                Stmt::from(GraphAttributes::rankdir(rankdir::TB)),
                Stmt::from(GraphAttributes::splines(true))
            ]
        };

        for id in self.node_indices() {
            let w = self.node_weight(id).unwrap();

            // Create graphviz node
            let mut gv_node = Node {
                id: NodeId(Id::Plain(id.index().to_string()), None),
                attributes: vec![
                    NodeAttributes::label(format!("\"{}\"", w).to_string())
                ],
            };

            // Add node attribute if it exists
            if node_attr.is_some() {
                let na = node_attr.unwrap();
                if na.contains_key(&id) {
                    gv_node.attributes.push(na.get(&id).unwrap().clone());
                }
            }

            g.add_stmt(Stmt::Node(gv_node));
        }

        for id in self.edge_indices() {
            let ep = self.edge_endpoints(id).unwrap();
            let w = self.edge_weight(id).unwrap();

            // Create graphviz edge
            let mut e = edge!(
                node_id!(ep.0.index().to_string()) =>
                node_id!(ep.1.index().to_string()));

            e.attributes.push(
                EdgeAttributes::label(format!("\"{}\"", w).to_string()));
            g.add_stmt(Stmt::Edge(e));
        }

        // Export to pdf
        let dot = g.print(&mut PrinterContext::new(true, 4, "\n".to_string(), 90));
        Ok(dot)
    }
}
