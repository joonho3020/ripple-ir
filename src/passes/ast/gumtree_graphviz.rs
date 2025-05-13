use graphviz_rust::{
    attributes::{rankdir, color_name, EdgeAttributes, GraphAttributes, NodeAttributes},
    exec_dot,
    dot_structures::*,
    dot_generator::{edge, node_id, id},
    cmd::{CommandArg, Format},
    printer::{DotPrinter, PrinterContext},
};
use indexmap::IndexMap;
use petgraph::graph::NodeIndex;
use petgraph::graph::Graph;
use petgraph::Direction::Outgoing;
use petgraph::Direction::Incoming;
use crate::{common::RippleIRErr, passes::ast::firrtlgraph::*};
use super::gumtree::*;

type TreeNodeIndex = NodeIndex;
type PrintNodeIndex = NodeIndex;
type PrintGraph = Graph<FirrtlNode, PrintGraphEdge>;

#[derive(Debug, Clone, Copy)]
enum PrintGraphEdge {
    SrcEdge,
    DstEdge,
    TopDownEdge,
    BottomUpEdge,
}

impl PrintGraphEdge {
    fn color(self: &Self) -> color_name {
        match self {
            Self::SrcEdge => { color_name::black },
            Self::DstEdge => { color_name::black },
            Self::TopDownEdge => { color_name::red },
            Self::BottomUpEdge => { color_name::blue },
        }
    }
}

impl GumTree {
    fn graphviz(
        pg: &PrintGraph,
        src_nodes: &Vec<NodeIndex>,
        dst_nodes: &Vec<NodeIndex>
    ) -> graphviz_rust::dot_structures::Graph {
        let mut g = graphviz_rust::dot_structures::Graph::DiGraph {
            id: graphviz_rust::dot_generator::id!(""),
            strict: true,
            stmts: vec![
                Stmt::from(GraphAttributes::rankdir(rankdir::TB))
            ]
        };

        // add invisible nodes to constrain placement
        let helper1 = "Helper1".to_string();
        let helper2 = "Helper2".to_string();

        g.add_stmt(Stmt::Node(Node {
            id: NodeId(Id::Plain(helper1.clone()), None),
            attributes: vec![
                NodeAttributes::style("invis".to_string())
            ],
        }));

        g.add_stmt(Stmt::Node(Node {
            id: NodeId(Id::Plain(helper2.clone()), None),
            attributes: vec![
                NodeAttributes::style("invis".to_string())
            ],
        }));

        let mut helper_edge = edge!(node_id!(helper1.clone()) => node_id!(helper2.clone()));
        helper_edge.attributes.push(EdgeAttributes::style("invis".to_string()));
        g.add_stmt(Stmt::Edge(helper_edge));

        // cluster of source AST
        let mut src_sg = graphviz_rust::dot_structures::Subgraph {
            id: Id::Plain("cluster_src_tree".to_string()),
            stmts: vec![]
        };

        for id in src_nodes.iter() {
            let node = pg.node_weight(*id).unwrap();

            let mut gv_node = Node {
                id: NodeId(Id::Plain(id.index().to_string()), None),
                attributes: vec![
                    NodeAttributes::label(format!("\"{}\"", node).to_string())
                ],
            };

            // Check if the node has been deleted from the source AST
            let mut has_match = false;
            for eref in pg.edges_directed(*id, Outgoing) {
                match eref.weight() {
                    PrintGraphEdge::TopDownEdge |
                        PrintGraphEdge::BottomUpEdge => {
                            has_match = true;
                        }
                    _ => { }
                }
            }

            if !has_match {
                gv_node.attributes.push(NodeAttributes::color(color_name::red));
            }

            src_sg.stmts.push(Stmt::Node(gv_node));

            // add constraints that will enforce rankings between the
            // source graph nodes and the helper node
            let mut helper_edge = edge!(
                node_id!(id.index().to_string()) =>
                node_id!(helper1.clone()));
            helper_edge.attributes.push(EdgeAttributes::style("invis".to_string()));
            g.add_stmt(Stmt::Edge(helper_edge));
        }

        g.add_stmt(Stmt::Subgraph(src_sg));

        // cluster of dst AST
        let mut dst_sg = graphviz_rust::dot_structures::Subgraph {
            id: Id::Plain("cluster_dst_tree".to_string()),
            stmts: vec![]
        };

        for id in dst_nodes.iter() {
            let node = pg.node_weight(*id).unwrap();
            let mut gv_node = Node {
                id: NodeId(Id::Plain(id.index().to_string()), None),
                attributes: vec![
                    NodeAttributes::label(format!("\"{}\"", node).to_string())
                ],
            };

            // Check if the node has been added to the dst AST
            let mut has_match = false;
            for eref in pg.edges_directed(*id, Incoming) {
                match eref.weight() {
                    PrintGraphEdge::TopDownEdge |
                        PrintGraphEdge::BottomUpEdge => {
                            has_match = true;
                        }
                    _ => { }
                }
            }

            if !has_match {
                gv_node.attributes.push(NodeAttributes::color(color_name::green));
            }

            dst_sg.stmts.push(Stmt::Node(gv_node));

            // add constraints that will enforce rankings between the
            // helper node and the dst graph nodes
            let mut helper_edge = edge!(
                node_id!(helper2.clone()) =>
                node_id!(id.index().to_string()));
            helper_edge.attributes.push(EdgeAttributes::style("invis".to_string()));
            g.add_stmt(Stmt::Edge(helper_edge));
        }

        g.add_stmt(Stmt::Subgraph(dst_sg));

        // add edges
        for eidx in pg.edge_indices() {
            let ep = pg.edge_endpoints(eidx).unwrap();
            let w = pg.edge_weight(eidx).unwrap();

            let mut e = edge!(node_id!(ep.0.index().to_string()) =>
                node_id!(ep.1.index().to_string()));
            e.attributes.push(EdgeAttributes::color(w.color()));
            match w {
                PrintGraphEdge::TopDownEdge |
                    PrintGraphEdge::BottomUpEdge => {
                        e.attributes.push(EdgeAttributes::constraint(false));
                    }
                _ => {
                }
            }
            g.add_stmt(Stmt::Edge(e));
        }
        return g;
    }

    fn add_graph(
        ln: &FirrtlGraph,
        print_graph: &mut PrintGraph,
        edge_type: PrintGraphEdge,
    ) -> IndexMap<TreeNodeIndex, PrintNodeIndex> {
        let mut src_map: IndexMap<TreeNodeIndex, PrintNodeIndex> = IndexMap::new();
        for idx in ln.graph.node_indices() {
            let w = ln.graph.node_weight(idx).unwrap();
            let print_idx = print_graph.add_node(w.clone());
            src_map.insert(idx, print_idx);
        }
        for idx in ln.graph.edge_indices() {
            let ep = ln.graph.edge_endpoints(idx).unwrap();
            print_graph.add_edge(
                *src_map.get(&ep.0).unwrap(),
                *src_map.get(&ep.1).unwrap(),
                edge_type);
        }
        return src_map;
    }


    pub fn export_gumtree_diff(
        &self,
        src: &FirrtlGraph,
        dst: &FirrtlGraph,
        path: &str
    ) -> Result<(), RippleIRErr> {
        let mut print_graph: PrintGraph = Graph::new();
        let src_map = Self::add_graph(src, &mut print_graph, PrintGraphEdge::SrcEdge);
        let dst_map = Self::add_graph(dst, &mut print_graph, PrintGraphEdge::DstEdge);

        let td_matches = self.top_down_phase(src, dst);
        for Match(src_id, dst_id) in td_matches.iter() {
            let p_src_id = src_map.get(src_id).unwrap();
            let p_dst_id = dst_map.get(dst_id).unwrap();
            print_graph.add_edge(*p_src_id, *p_dst_id, PrintGraphEdge::TopDownEdge);
        }

        println!("========= Top down matches ==============");
        for Match(src_id, dst_id) in td_matches.iter() {
            println!("---------------------------------------");
            println!("{:?}\n{:?}",
                src.graph.node_weight(*src_id).unwrap(),
                dst.graph.node_weight(*dst_id).unwrap());
        }


        let bu_matches = self.bottom_up_phase(src, dst, &td_matches);
        for Match(src_id, dst_id) in bu_matches.iter() {
            let p_src_id = src_map.get(src_id).unwrap();
            let p_dst_id = dst_map.get(dst_id).unwrap();
            print_graph.add_edge(*p_src_id, *p_dst_id, PrintGraphEdge::BottomUpEdge);
        }

        println!("========= Bottom up matches ==============");
        for Match(src_id, dst_id) in bu_matches.iter() {
            println!("---------------------------------------");
            println!("{:?}\n{:?}",
                src.graph.node_weight(*src_id).unwrap(),
                dst.graph.node_weight(*dst_id).unwrap());
        }

        let src_print_nodes: Vec<NodeIndex> = src_map.iter().map(|(_, v)| *v).collect();
        let dst_print_nodes: Vec<NodeIndex> = dst_map.iter().map(|(_, v)| *v).collect();

        let g = Self::graphviz(&print_graph, &src_print_nodes, &dst_print_nodes);
        let dot = g.print(&mut PrinterContext::default());
        exec_dot(dot, vec![Format::Pdf.into(), CommandArg::Output(path.to_string())])?;

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use chirrtl_parser::parse_circuit;
    use test_case::test_case;

    use super::*;
    use crate::passes::ast::firrtlgraph::FirrtlGraph;
    use crate::common::RippleIRErr;

    #[test_case("GCD", "GCDDelta" ; "GCD")]
    #[test_case("Adder", "Subtracter" ; "Adder")]
    fn run(src: &str, dst: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(format!("./test-inputs/{}.fir", src))?;
        let circuit = parse_circuit(&source).expect("firrtl parser");
        let src_graph = FirrtlGraph::from_circuit(&circuit);

        let source_delta = std::fs::read_to_string(format!("./test-inputs/{}.fir", dst))?;
        let circuit_delta = parse_circuit(&source_delta).expect("firrtl parser");
        let dst_graph = FirrtlGraph::from_circuit(&circuit_delta);

        let gumtree = GumTree::default();
        gumtree.export_gumtree_diff(&src_graph, &dst_graph, &format!("./test-outputs/{}.diff.pdf", src))?;

        Ok(())
    }


    #[test_case("GCD", "GCDDelta" ; "GCD")]
    #[test_case("AESStep1", "AESStep2" ; "AES1")]
    #[test_case("AESStep2", "AESStep3" ; "AES2")]
    #[test_case("CordicStep1", "CordicStep2" ; "Cordic1")]
    #[test_case("CordicStep2", "CordicStep3" ; "Cordic2")]
    #[test_case("FFTStep1", "FFTStep2" ; "FFT1")]
    #[test_case("FFTStep2", "FFTStep3" ; "FFT2")]
    #[test_case("FFTStep3", "FFTStep4" ; "FFT3")]
    fn compute_overlap(src: &str, dst: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(format!("./test-inputs/{}.fir", src))?;
        let circuit = parse_circuit(&source).expect("firrtl parser");
        let src_graph = FirrtlGraph::from_circuit(&circuit);

        let source_delta = std::fs::read_to_string(format!("./test-inputs/{}.fir", dst))?;
        let circuit_delta = parse_circuit(&source_delta).expect("firrtl parser");
        let dst_graph = FirrtlGraph::from_circuit(&circuit_delta);

        let gumtree = GumTree::default();
        let matches = gumtree.diff(&src_graph, &dst_graph);

        let src_graph_sz = src_graph.graph.node_count();
        let match_sz = matches.len();

        let percentage = match_sz as f32 / src_graph_sz as f32 * 100.0;
        println!("percentage for {:?} vs {:?}: match {} / src {} x 100 = {} %",
            src, dst,
            match_sz, src_graph_sz, percentage);

        Ok(())
    }
}
