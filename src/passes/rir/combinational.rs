use std::collections::VecDeque;

use rusty_firrtl::Identifier;
use petgraph::graph::NodeIndex;
use petgraph::Direction::Incoming;

use crate::ir::rir::{rgraph::RippleGraph ,rir::RippleIR, rnode::RippleNodeType};
use indexmap::IndexMap;
use indexmap::IndexSet;

/// List of input ports that are combinationally coupled to each output port
pub type PortCombDeps = IndexMap<Identifier, IndexSet<Identifier>>;

/// Maps each module to PortCombDeps
pub type HierCombDeps = IndexMap<Identifier, PortCombDeps>;

/// Performs combinational dependency analaysis
pub fn combinational_analaysis(rir: &RippleIR) -> HierCombDeps {
    let mut ret = HierCombDeps::new();
    for hier in rir.hierarchy.topo_order() {
        let rg = rir.graphs.get(hier.name()).unwrap();
        combinational_analaysis_graph(rg, hier.name(), &mut ret);
    }
    ret
}

fn combinational_analaysis_graph(rg: &RippleGraph, name: &Identifier, deps: &mut HierCombDeps) {
    let graph = rg.flat_graph();

    let mut output_ids: Vec<NodeIndex> = vec![];
    for id in graph.node_indices() {
        let node = graph.node_weight(id).unwrap();
        if node.data.tpe == RippleNodeType::Output {
            output_ids.push(id);
        }
    }

    let mut port_deps = PortCombDeps::new();
    for id in output_ids {
        let oport_name = graph.node_weight(id).unwrap().data.name.as_ref().unwrap();
        let dep_ids = find_comb_cone(rg, id, deps);

        port_deps.insert(oport_name.clone(), IndexSet::new());

        for dep_id in dep_ids {
            let dep = graph.node_weight(dep_id).unwrap();
            if dep.data.tpe == RippleNodeType::Input {
                let iport_name = dep.data.name.as_ref().unwrap().clone();
                port_deps.entry(oport_name.clone()).or_insert_with(IndexSet::new).insert(iport_name);
            }
        }
    }

    deps.insert(name.clone(), port_deps);
}

fn find_comb_cone(rg: &RippleGraph, id: NodeIndex, deps: &HierCombDeps) -> Vec<NodeIndex> {
    let graph = rg.flat_graph();
    let mut ret: Vec<NodeIndex> = vec![];
    let mut q: VecDeque<NodeIndex> = VecDeque::new();
    q.push_back(id);

    while !q.is_empty() {
        let id = q.pop_front().unwrap();
        for pid in graph.neighbors_directed(id, Incoming) {
            let parent = graph.node_weight(pid).unwrap();
            match &parent.data.tpe {
                RippleNodeType::Reg |
                    RippleNodeType::CMem |
                    RippleNodeType::RegReset |
                    RippleNodeType::SMem(..) |
                    RippleNodeType::ReadMemPort(..) |
                    RippleNodeType::WriteMemPort(..) |
                    RippleNodeType::InferMemPort(..) |
                    RippleNodeType::Printf(..) |
                    RippleNodeType::Assert(..) => {
                        // End of combinational cone
                    continue;
                }
                RippleNodeType::Inst(module) => {
                    let cur_deps = deps.get(module).unwrap();
                    let parent_str = parent.data.name.as_ref().unwrap().to_string();
                    let mut iports = vec![];

                    for (port_name, coupled_ports) in cur_deps.iter() {
                        if parent_str.contains(&port_name.to_string()) {
                            iports.extend(coupled_ports.iter().cloned());
                        }
                    }

                    let agg_id = rg.find_corresp_agg_id(parent.id);
                    let inst_port_ids = rg.flatids_under_agg(agg_id);
                    for inst_port_id in inst_port_ids {
                        let port = graph.node_weight(inst_port_id).unwrap();
                        let port_name = port.data.name.as_ref().unwrap().to_string();

                        for ip in iports.iter() {
                            if port_name.contains(&ip.to_string()) {
                                q.push_back(inst_port_id);
                                ret.push(inst_port_id);
                            }
                        }
                    }
                }
                _ => {
                    ret.push(pid);
                    q.push_back(pid);
                }
            }
        }
    }
    ret
}

#[cfg(test)]
pub mod test {
    use test_case::test_case;
    use crate::common::RippleIRErr;
    use crate::passes::runner::run_passes_from_filepath;
    use crate::passes::rir::from_fir::from_fir;
    use super::*;

    #[test_case("GCD" ; "GCD")]
    #[test_case("Fir" ; "Fir")]
    #[test_case("Hierarchy" ; "Hierarchy")]
    #[test_case("CombHierarchy" ; "CombHierarchy")]
    fn run(input: &str) -> Result<(), RippleIRErr> {
        let filename = format!("./test-inputs/{}.fir", input);

        let fir = run_passes_from_filepath(&filename)?;
// fir.export("./test-outputs", "comb.fir")?;

        let rir = from_fir(&fir);
// rir.export("./test-outputs", "comb.rir")?;

        let ret = combinational_analaysis(&rir);
        println!("Deps {:?}", ret);

        Ok(())
    }
}
