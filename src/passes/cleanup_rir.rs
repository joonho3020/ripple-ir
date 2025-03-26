use petgraph::{graph::NodeIndex, Direction::Outgoing};
use indexmap::IndexMap;
use crate::ir::{typetree::GroundType, *};

/// Cleanup
/// - Array and memory nodes
/// - Remove phi selection signals that are irrelevant
pub fn cleanup_rir(rir: &mut RippleIR) {
    for (_name, rg) in rir.graphs.iter_mut() {
        cleanup_rg_memory(rg);
    }
}

fn cleanup_rg_memory(rg: &mut RippleGraph) {
    let mut node_map: IndexMap<RippleNode, AggNodeIndex> = IndexMap::new();

    for agg_id in rg.node_indices_agg() {
        let node_agg = rg.node_weight_agg(agg_id);

        let ttree = node_agg.0.unwrap();
        let all_leaves = ttree.view().unwrap().leaves();

        let mut all_graph_nodes: Vec<NodeIndex> = all_leaves.iter().map(|id| {
            *rg.flatid(agg_id, *id).unwrap()
        }).collect();
        all_graph_nodes.sort();

        match node_agg.1.unwrap().nt {
            RippleNodeType::SMem(..) |
            RippleNodeType::CMem => {
                let gt = if node_agg.1.unwrap().nt == RippleNodeType::CMem {
                    GroundType::CMem
                } else {
                    GroundType::SMem
                };
                let node = RippleNode::new(
                        Some(node_agg.1.unwrap().name.clone()),
                        node_agg.1.unwrap().nt.clone(),
                        gt);

                // Add new node to the graph
                let id = rg.graph.add_node(node.clone());
                node_map.insert(node, agg_id);

                // Add single memory node
                rg.flatid_aggleaf_bimap.insert(id, AggNodeLeafIndex::new(agg_id, all_leaves));

                // Connect memory node to its ports
                let agg_edges = rg.edges_directed_agg(agg_id, Outgoing);
                for (_edge_identifier, edges) in agg_edges.iter() {
                    for eid in edges {
                        let dst = rg.graph.edge_endpoints(*eid).unwrap().1;
                        let ew = rg.graph.edge_weight(*eid).unwrap();
                        rg.add_edge(id, dst, ew.clone());
                    }
                }

                // Remove existing nodes
                for id in all_graph_nodes.iter().rev() {
                    rg.remove_node(*id);
                }
            }
            _ => { }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::common::RippleIRErr;
    use crate::passes::runner::run_passes_from_filepath;
    use crate::common::graphviz::GraphViz;
    use crate::passes::from_fir::from_fir;
    use super::*;

    fn run_simple(input: &str) -> Result<(), RippleIRErr> {
        let fir = run_passes_from_filepath(input)?;
        for (module, fg) in fir.graphs.iter() {
            fg.export_graphviz(&format!("./test-outputs/{}-{}.fir.pdf",
                    fir.name.to_string(), module.to_string()), None, None, false)?;
        }

        let mut rir = from_fir(&fir);
        for (module, rg) in rir.graphs.iter() {
            rg.export_graphviz(&format!("./test-outputs/{}-{}.rir.pdf",
                    rir.name.to_string(), module.to_string()), None, None, false)?;
        }

        cleanup_rir(&mut rir);
        for (module, rg) in rir.graphs.iter() {
            rg.export_graphviz(&format!("./test-outputs/{}-{}.rir.cleanup.pdf",
                    rir.name.to_string(), module.to_string()), None, None, false)?;
        }
        Ok(())
    }

    #[test]
    fn singleportsram() {
        run_simple("./test-inputs/SinglePortSRAM.fir")
            .expect("singleportsram");
    }
}
