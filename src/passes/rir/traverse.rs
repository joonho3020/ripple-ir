use crate::common::RippleIRErr;
use crate::ir::rir::rir::RippleIR;
use crate::ir::rir::rnode::*;
use crate::ir::rir::rgraph::RippleGraph;
use crate::ir::rir::agg::*;
use crate::timeit;
use crate::common::graphviz::*;
use std::collections::VecDeque;
use rusty_firrtl::*;
use graphviz_rust::attributes::{NodeAttributes, color_name};

pub fn traverse_aggregate(rir: RippleIR, export: bool) -> Result<(), RippleIRErr> {
    timeit!("Aggregate Traversal", {
        for (module, rg) in rir.graphs.iter() {
            traverse_graph_aggregate(module, rg, export)?;
        }
    });

    Ok(())
}

fn traverse_graph_aggregate(module: &Identifier, rg: &RippleGraph, export: bool) -> Result<(), RippleIRErr> {
    let mut q: VecDeque<AggNodeIndex> = VecDeque::new();

    for agg_id in rg.node_indices_agg().iter() {
        let agg_w = rg.node_weight_agg(*agg_id).unwrap();

        // Collect all the IOs
        match agg_w.nt {
            RippleNodeType::Input |
                RippleNodeType::Output |
                RippleNodeType::UIntLiteral(..) |
                RippleNodeType::SIntLiteral(..) |
                RippleNodeType::DontCare |
                RippleNodeType::CMem |
                RippleNodeType::SMem(..) |
                RippleNodeType::Inst(..) => {
                    q.push_back(*agg_id);
                }
            _ => {
            }
        }
    }

    let mut file_names: Vec<String> = vec![];
    let mut iter_outer = 0;
    let mut vis_map = rg.vismap_agg();

    while !q.is_empty() {
        let agg_id = q.pop_front().unwrap();
        if vis_map.is_visited(agg_id) {
            continue;
        }
        vis_map.visit(agg_id);

        let mut node_attributes = NodeAttributeMap::default();
        let node_ids = rg.flatids_under_agg(agg_id);
        for id in node_ids {
            node_attributes.insert(id, NodeAttributes::color(color_name::green));
        }

        let agg_edges = rg.edges_agg(agg_id);
        for (iter_inner, agg_edge) in agg_edges.iter().enumerate() {
            if export {
                let mut cur_node_attributes = node_attributes.clone();

                let dst_node_ids = rg.flatids_under_agg(agg_edge.dst);
                for id in dst_node_ids {
                    cur_node_attributes.insert(id, NodeAttributes::color(color_name::blue));
                }

                let mut edge_attributes = EdgeAttributeMap::default();
                let edge_ids = rg.flatedges_under_agg(agg_id, agg_edge);
                for eid in edge_ids {
                    edge_attributes.insert(eid, NodeAttributes::color(color_name::red));
                }

                let out_name = format!(
                    "./test-outputs/{}-{}-{}.traverse.agg.pdf",
                    module, iter_outer, iter_inner);

                rg.export_graphviz(
                    &out_name,
                    Some(cur_node_attributes).as_ref(),
                    Some(edge_attributes).as_ref(),
                    false)?;
                file_names.push(out_name);
            }
            if !vis_map.is_visited(agg_edge.dst) {
                q.push_back(agg_edge.dst);
            }
        }
        iter_outer += 1;
    }

    // dead code
    if vis_map.has_unvisited() {
        let unvisited = vis_map.unvisited_ids();
        for agg_id in unvisited {
            let agg_edges = rg.edges_agg(agg_id);
            let agg_node = rg.node_weight_agg(agg_id).unwrap();
            let ttree = agg_node.ttree.as_ref().unwrap().view().unwrap();
            if agg_edges.len() != 0 && !ttree.is_empty() {
                panic!("Aggregate traversal missed agg_node {:?} with {:?} agg_edges",
                    agg_node, agg_edges.len());
            }
        }
    }

    if export {
        rg.create_gif(
            &format!("./test-outputs/{}.gif", module),
            &file_names)?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::common::RippleIRErr;
    use crate::passes::runner::run_passes_from_filepath;
    use crate::passes::rir::from_fir::from_fir;
    use super::*;

    fn run_traverse(input: &str, export: bool) -> Result<(), RippleIRErr> {
        let fir = run_passes_from_filepath(input)?;
        let rir = from_fir(&fir);
        traverse_aggregate(rir, export)?;
        Ok(())
    }
    #[test]
    fn traverse_gcd() {
        run_traverse("./test-inputs/GCD.fir", false)
            .expect("gcd traverse assumption");
    }

    #[test]
    fn traverse_decoupledmux() {
        run_traverse("./test-inputs/DecoupledMux.fir", false)
            .expect("decoupledmux traverse assumption");
    }

    #[test]
    fn traverse_singleportsram() {
        run_traverse("./test-inputs/SinglePortSRAM.fir", false)
            .expect("singleportsram traverse assumption");
    }

    #[test]
    fn traverse_aggregatesram() {
        run_traverse("./test-inputs/AggregateSRAM.fir", false)
            .expect("aggregatesram traverse assumption");
    }

// #[test]
// fn traverse_rocket() {
// run_traverse("./test-inputs/chipyard.harness.TestHarness.RocketConfig.fir", false)
// .expect("rocket traverse assumption");
// }

// #[test]
// fn traverse_boom() {
// run_traverse("./test-inputs/chipyard.harness.TestHarness.LargeBoomV3Config.fir", false)
// .expect("boom traverse assumption");
// }
}
