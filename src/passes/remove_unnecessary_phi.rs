use crate::ir::*;
use petgraph::{graph::{NodeIndex, EdgeIndex}, visit::EdgeRef, Direction::{Incoming, Outgoing}};

pub fn remove_unnecessary_phi(ir: &mut RippleIR) {
    for (_id, rg) in ir.graphs.iter_mut() {
        remove_unnecessary_phi_in_ripple_graph(rg);
    }
}

fn remove_unnecessary_phi_in_ripple_graph(rg: &mut RippleGraph) {
    let mut remove_nodes: Vec<NodeIndex> = vec![];
    for id in rg.graph.node_indices() {
        let node = rg.graph.node_weight(id).unwrap();
        match node {
            NodeType::Phi(_) => {
                if is_removable(rg, id) {
                    connect_phi_parent_to_child(rg, id);
                    remove_nodes.push(id);
                }
            }
            _ => {
                continue;
            }
        }
    }

    for id in remove_nodes {
        rg.graph.remove_node(id);
    }
}

/// Remove phi nodes when
/// - There is no selection signal
/// - The selection signal is always true
fn is_removable(rg: &RippleGraph, id: NodeIndex) -> bool {
    let mut has_sel = false;
    let mut has_non_trivial_sel = false;

    let pedges = rg.graph.edges_directed(id, Incoming);
    for pedge in pedges {
        let edge = rg.graph.edge_weight(pedge.id()).unwrap();
        match edge {
            EdgeType::PhiSel(_sel) => {
                has_sel = true;
            }
            EdgeType::PhiInput(_prior, cond, _sink, _source) => {
                if !has_non_trivial_sel {
                    has_non_trivial_sel = !cond.always_true()
                }
            }
            EdgeType::DontCare(_) => {
            }
            _ => {
                panic!("Unrecognized driving edge {:?} for phi node", edge);
            }
        }
    }

    if has_sel && !has_non_trivial_sel {
        let pedges = rg.graph.edges_directed(id, Incoming);
        for pedge in pedges {
            let edge = rg.graph.edge_weight(pedge.id()).unwrap();
            eprintln!("{:?}", edge);
        }
        panic!("Phi node has incoming sel, but only has trivial selectors");
    }

    if !has_sel || !has_non_trivial_sel {
        return true;
    } else {
        return false;
    }
}

fn connect_phi_parent_to_child(rg: &mut RippleGraph, id: NodeIndex) {
    let childs: Vec<NodeIndex> = rg.graph.neighbors_directed(id, Outgoing).into_iter().collect();
    if childs.len() == 0 {
        return;
    }

    assert!(childs.len() == 1, "Phi node is driving multiple nodes {}", childs.len());

    let pedges: Vec<EdgeIndex> = rg.graph.edges_directed(id, Incoming).into_iter().map(|x| x.id()).collect();
    for peid in pedges.iter() {
        let ew = rg.graph.edge_weight(*peid).unwrap();
        let ep = rg.graph.edge_endpoints(*peid).unwrap();
        let src = ep.0;
        match ew {
            EdgeType::PhiInput(_, _, r, e) => {
                rg.graph.add_edge(src, childs[0], EdgeType::Wire(r.clone(), e.clone()));
            }
            EdgeType::DontCare(_) => {
                rg.graph.add_edge(src, childs[0], ew.clone());
            }
            _ => {
                panic!("Phi node driver edge should be PhiInput, got {:?}", ew);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        common::graphviz::GraphViz,
        passes::from_ast::{from_circuit, from_circuit_module}
    };
    use chirrtl_parser::parse_circuit;

    /// Run the AST to graph conversion and export the graph form
    fn run(name: &str) -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string(format!("./test-inputs/{}.fir", name))?;
        let circuit = parse_circuit(&source).expect("firrtl parser");

        let mut ir = from_circuit(&circuit);
        remove_unnecessary_phi(&mut ir);
        for (sub_name, graph) in ir.graphs {
            graph.export_graphviz(&format!("./test-outputs/{}-{}.remove_phi.dot.pdf", name, sub_name), None, true)?;
        }
        Ok(())
    }

    #[test]
    fn gcd() {
        run("GCD").expect("GCD");
    }

    #[test]
    fn nestedwhen() {
        run("NestedWhen").expect("NestedWhen");
    }

    #[test]
    fn nestedbundle() {
        run("NestedBundle").expect("NestedBundle");
    }

    #[test]
    fn singleport_sram() {
        run("SinglePortSRAM").expect("SinglePortSRAM");
    }

    #[test]
    fn hierarchy() {
        run("Hierarchy").expect("Hierarchy");
    }

    use chirrtl_parser::lexer::FIRRTLLexer;
    use chirrtl_parser::firrtl::CircuitModuleParser;

    /// Check of the AST to graph conversion works for each CircuitModule
    fn run_check_completion(input_dir: &str) -> Result<(), std::io::Error> {
        for entry in std::fs::read_dir(input_dir)? {
            let entry = entry?;
            let path = entry.path();

            // Check if it's a file (not a directory)
            if path.is_file() {
                match std::fs::read_to_string(&path) {
                    Ok(source) => {
                        let lexer = FIRRTLLexer::new(&source);
                        let parser = CircuitModuleParser::new();

                        println!("Parsing file: {:?}", path);
                        let ast = parser.parse(lexer).expect("TOWORK");
                        let (_, mut rg) = from_circuit_module(&ast);
                        remove_unnecessary_phi_in_ripple_graph(&mut rg);
                    }
                    Err(e) => {
                        eprintln!("Could not read file {}: {}", path.display(), e);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn rocket_check_completion() {
        run_check_completion("./test-inputs/rocket-modules/")
            .expect("rocket conversion failed");
    }

    #[test]
    fn boom_check_completion() {
        run_check_completion("./test-inputs/boom-modules/")
            .expect("boom conversion failed");
    }
}
