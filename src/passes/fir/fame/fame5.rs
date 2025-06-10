use petgraph::visit::EdgeRef;
use petgraph::Direction::Incoming;
use rusty_firrtl::{Expr, Reference};
use rusty_firrtl::{Annotations, Identifier};
use serde_json::Value;
use indexmap::IndexMap;
use std::collections::HashMap;
use std::collections::HashSet;
use petgraph::graph::NodeIndex;
use crate::common::graphviz::DefaultGraphVizCore;
use crate::ir::{fir::*, hierarchy::InstPath};
use crate::passes::fir::fame::log2_ceil;
use crate::passes::fir::fame::uint_ttree;


type InstModuleMap = IndexMap<Identifier, Identifier>;

pub fn fame5_transform(fir: &mut FirIR) {
    let inst_module_map = get_fame5_inst_to_module_map(&find_fame5_target_inst_paths(&fir.annos));
    println!("inst_module_map {:?}", inst_module_map);

    let (insts, mod_to_fame5_opt) = module_to_fame5(&inst_module_map);
    println!("Instances {:?}", insts);
    println!("Module {:?}", mod_to_fame5_opt);

    if mod_to_fame5_opt.is_none() {
        return;
    }

    let mod_to_fame5 = mod_to_fame5_opt.unwrap();
    let mod_to_fame5_hier_id = fir.hier.id(mod_to_fame5).unwrap();
    let parents = fir.hier.graph.neighbors_directed(mod_to_fame5_hier_id, Incoming);
    assert!(parents.clone().count() == 1, "Module to FAME5 has multiple parent modules");

    let parent_hier_id = parents.into_iter().next().unwrap();
    let parent_module = fir.hier.node_weight(parent_hier_id).unwrap().name();
    let parent_fg = fir.graphs.get_mut(parent_module).unwrap();

    let (top, hostclock) = find_hostclock(&fir.annos);
    assert!(&Identifier::Name(top.module) == parent_module);

    let (top, hostreset) = find_hostreset(&fir.annos);
    assert!(&Identifier::Name(top.module) == parent_module);

    update_fame5_top(parent_fg, &hostclock, &hostreset, mod_to_fame5);

}

type TargetInstPathString = String;

fn find_class_target_annos<'a>(annos: &'a Annotations, class: &str) -> Vec<&'a TargetInstPathString> {
    let mut ret = vec![];
    if let Some(annos_list) = annos.0.as_array() {
        for anno in annos_list.iter() {
            if let Some(map) = anno.as_object() {
                if let Some(Value::String(class_value)) = map.get("class") {
                    if class_value == class {
                        if let Some(Value::String(target_value)) = map.get("target") {
                            ret.push(target_value);
                        }
                    }
                }
            }
        }
    }
    ret
}

fn find_fame5_target_inst_paths(annos: &Annotations) -> Vec<&TargetInstPathString> {
    find_class_target_annos(annos, "midas.targetutils.FirrtlEnableModelMultiThreadingAnnotation")
}

fn get_fame5_inst_to_module_map(targets: &Vec<&TargetInstPathString>) -> InstModuleMap  {
    let mut inst_to_module: InstModuleMap = InstModuleMap::new();
    for target in targets {
        let parts: Vec<&str> = target.split(|c| c == '|' || c == '>').collect();
        assert!(parts.len() >= 2);

        let inst_paths = InstPath::parse_inst_hierarchy_path(parts[1]);
        let inst_path_leaf = inst_paths.last().unwrap();
        inst_to_module.insert(
            Identifier::Name(inst_path_leaf.inst.as_ref().expect("No inst in FAME5 anno").clone()),
            Identifier::Name(inst_path_leaf.module.clone()));
    }
    inst_to_module
}

fn find_hostclock_or_reset(annos: &Annotations, class: &str) -> (InstPath, Identifier) {
    let host_annos = find_class_target_annos(annos, class);
    assert!(host_annos.len() == 1);

    for host_anno in host_annos {
        let parts: Vec<&str> = host_anno.split(|c| c == '|' || c == '>').collect();
        assert!(parts.len() == 3);

        let inst_paths = InstPath::parse_inst_hierarchy_path(parts[1]);
        let inst_path_leaf = inst_paths.last().expect("Could not find FAME host clock/reset");
        return (inst_path_leaf.clone(), Identifier::Name(parts[2].to_string()));
    }
    unreachable!();
}

fn find_hostclock(annos: &Annotations) -> (InstPath, Identifier) {
    find_hostclock_or_reset(annos, "midas.passes.fame.FAMEHostClock")
}

fn find_hostreset(annos: &Annotations) -> (InstPath, Identifier) {
    find_hostclock_or_reset(annos, "midas.passes.fame.FAMEHostReset")
}

fn module_to_fame5(inst_to_module: &InstModuleMap) -> (Vec<&Identifier>, Option<&Identifier>) {
   let mut per_module_inst_count: HashMap<&Identifier, usize> = HashMap::new();
    for value in inst_to_module.values() {
        *per_module_inst_count.entry(value).or_insert(0) += 1;
    }

    let modules_with_multiple_insts: HashSet<&Identifier> = per_module_inst_count
        .iter()
        .filter(|(_, &count)| count > 1)
        .map(|(val, _)| *val)
        .collect();

    assert!(
        modules_with_multiple_insts.len() <= 1,
        "More than one module to perform FAME5"
    );

    if let Some(dup_val) = modules_with_multiple_insts.iter().next() {
        let insts = inst_to_module.iter()
            .filter(|(_, v)| v == dup_val)
            .map(|(k, _)| k)
            .collect();

        (insts, Some(modules_with_multiple_insts.iter().next().unwrap()))
    } else {
        (vec![], None)
    }
}

fn update_fame5_top(
    top: &mut FirGraph,
    host_clock: &Identifier,
    host_reset: &Identifier,
    module_to_fame5: &Identifier
) {
    let inst_ids: Vec<NodeIndex> = top.graph.node_indices().into_iter()
        .map(|id| {
            let node = top.graph.node_weight(id).unwrap();
            if let FirNodeType::Inst(module) = &node.nt {
                if module == module_to_fame5 {
                    return (true, id);
                }
            }
            return (false, id);
        })
        .filter(|x| x.0)
        .map(|x| x.1).collect();

    let host_clock_id = top.graph.node_indices().into_iter()
        .map(|id| {
            let node = top.graph.node_weight(id).unwrap();
            if node.name.is_some() && node.name.as_ref().unwrap() == host_clock {
                return (true, id);
            }
            return (false, id);
        })
        .filter(|x| x.0)
        .map(|x| x.1)
        .last()
        .expect(&format!("No host clock node {:?} found", host_clock));

    let host_reset_id = top.graph.node_indices().into_iter()
        .map(|id| {
            let node = top.graph.node_weight(id).unwrap();
            if node.name.is_some() && node.name.as_ref().unwrap() == host_reset {
                return (true, id);
            }
            return (false, id);
        })
        .filter(|x| x.0)
        .map(|x| x.1)
        .last()
        .expect(&format!("No host reset node {:?} found", host_reset));

    let nthreads = inst_ids.len() as u32;
    let thread_idx_id = add_thread_idx_reg(top, nthreads);
    connect_to_host_clock_and_reset(top, host_clock, host_reset, thread_idx_id);

    let fame5_inst_name = Identifier::Name(format!("{}_fame5", module_to_fame5));
    let fame5_mod_name  = Identifier::Name(format!("{}_fame5", module_to_fame5));
    assert!(!top.namespace.contains(&fame5_inst_name));

    let fame5_inst_id = top.graph.add_node(
        FirNode::new(
            Some(fame5_inst_name),
            FirNodeType::Inst(fame5_mod_name),
            None));
    connect_to_host_clock_and_reset(top, host_clock, host_reset, fame5_inst_id);

    let mut input_channel_map: IndexMap<&Identifier, Vec<(u32, NodeIndex)>> = IndexMap::new();
    for (idx, &inst_id) in inst_ids.iter().enumerate() {
        for eid in top.graph.edges_directed(inst_id, Incoming) {
            let edge = top.graph.edge_weight(eid.id()).unwrap();
            let ep = top.graph.edge_endpoints(eid.id()).unwrap();
            let dst_ref = edge.dst.as_ref().expect("Channel destination reference");
            match dst_ref {
                Reference::RefDot(_, input_channel_name) => {
                    if !input_channel_map.contains_key(input_channel_name) {
                        input_channel_map.insert(input_channel_name, vec![]);
                    }
                    input_channel_map.get_mut(input_channel_name).unwrap().push((idx as u32, ep.1));

                }
                _ => {
                    unreachable!();
                }
            }
        }
    }

}

fn find_host_clock_or_reset_id(fg: &FirGraph, host_node: &Identifier) -> NodeIndex {
    fg.graph.node_indices().into_iter()
        .map(|id| {
            let node = fg.graph.node_weight(id).unwrap();
            if node.name.is_some() && node.name.as_ref().unwrap() == host_node {
                return (true, id);
            }
            return (false, id);
        })
        .filter(|x| x.0)
        .map(|x| x.1)
        .last()
        .expect(&format!("No host node {:?} found", host_node))
}

fn add_thread_idx_reg(fg: &mut FirGraph, nthreads: u32) -> NodeIndex {
    let thread_idx_bits = log2_ceil(nthreads + 1);
    let thread_idx_name = Identifier::Name("threadIdx".to_string());
    let thread_idx_id = fg.graph.add_node(FirNode::new(
            Some(thread_idx_name.clone()),
            FirNodeType::RegReset,
            Some(uint_ttree(thread_idx_bits))));


    thread_idx_id
}
fn connect_to_host_clock_and_reset(
    fg: &mut FirGraph,
    host_clock: &Identifier,
    host_reset: &Identifier,
    dst_id: NodeIndex
)
{
    let host_clock_id = find_host_clock_or_reset_id(fg, host_clock);
    let host_reset_id = find_host_clock_or_reset_id(fg, host_reset);
    let dst = fg.graph.node_weight(dst_id).unwrap();
    let name = dst.name.as_ref().unwrap().clone();

    let hc_edge = FirEdge::new(
        Expr::Reference(Reference::Ref(host_clock.clone())),
        Some(Reference::Ref(name.clone())),
        FirEdgeType::Clock);
    fg.graph.add_edge(host_clock_id, dst_id, hc_edge);

    let hr_edge = FirEdge::new(
        Expr::Reference(Reference::Ref(host_reset.clone())),
        Some(Reference::Ref(name.clone())),
        FirEdgeType::Reset);
    fg.graph.add_edge(host_reset_id, dst_id, hr_edge);
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::{common::{read_annos, RippleIRErr}, passes::{ast::firrtl3_split_exprs::firrtl3_split_exprs, runner::run_fir_passes_from_circuit}};
    use test_case::test_case;
    use firrtl3_parser::parse_circuit as parse_firrtl3;

    #[test_case("FireSimGCD"; "FireSimGCD")]
    fn fame5(name: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(format!("./test-inputs-firrtl3/{}.fir", name)).expect("to_exist");
        let mut circuit = parse_firrtl3(&source).expect("firrtl parser");
        circuit.annos = read_annos(&format!("./test-inputs-firrtl3/{}.json", name))?;

        firrtl3_split_exprs(&mut circuit);
        let mut ir = run_fir_passes_from_circuit(&circuit)?;

        fame5_transform(&mut ir);

        Ok(())
    }
}
