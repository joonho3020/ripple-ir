use crate::ir::typetree::tnode::*;
use crate::ir::rir::rnode::*;
use crate::ir::rir::rgraph::*;
use crate::ir::rir::rir::*;

/// Cleanup
/// - Array and memory nodes
/// - Remove phi selection signals that are irrelevant
pub fn cleanup_rir(rir: &mut RippleIR) {
    for (_name, rg) in rir.graphs.iter_mut() {
        cleanup_rg_memory(rg);
    }
}

/// Remove all flat nodes that represent the memory node array and
/// replace it with a single node
fn cleanup_rg_memory(rg: &mut RippleGraph) {
    for agg_id in rg.node_indices_agg() {
        let node_agg = rg.node_weight_agg(agg_id).unwrap();
        match node_agg.nt {
            RippleNodeType::SMem(..) |
            RippleNodeType::CMem => {
                let gt = if node_agg.nt == RippleNodeType::CMem {
                    GroundType::CMem
                } else {
                    GroundType::SMem
                };
                let node = RippleNodeData::new(
                        Some(node_agg.name.clone()),
                        node_agg.nt.clone(),
                        gt);

                rg.merge_nodes_array_agg(agg_id, node);
            }
            _ => { }
        }
    }
}

/// Similar to cleanup_rg_memory except that this is for combinational memory arrays.
/// Possible when:
/// - All incoming edges are of type RippleEdgeType::ArrayAddr
fn cleanup_rg_array(rg: &mut RippleGraph) {
    for agg_id in rg.node_indices_agg() {
        let node_agg = rg.node_weight_agg(agg_id);
// let agg_edges = rg.edges_agg()
    }
}

#[cfg(test)]
mod test {
    use crate::common::RippleIRErr;
    use crate::passes::rir::traverse::traverse_aggregate;
    use crate::passes::runner::run_passes_from_filepath;
    use crate::common::graphviz::GraphViz;
    use crate::passes::rir::from_fir::from_fir;
    use super::*;

    fn run_simple(input: &str) -> Result<(), RippleIRErr> {
        let fir = run_passes_from_filepath(input)?;
        let mut rir = from_fir(&fir);
        cleanup_rir(&mut rir);

        for (module, rg) in rir.graphs.iter() {
            rg.export_graphviz(&format!("./test-outputs/{}-{}.rir.cleanup.pdf",
                    rir.name.to_string(), module.to_string()), None, None, false)?;
        }

        traverse_aggregate(rir, true)?;

        Ok(())
    }

    #[test]
    fn singleportsram() {
        run_simple("./test-inputs/SinglePortSRAM.fir")
            .expect("singleportsram");
    }
}
