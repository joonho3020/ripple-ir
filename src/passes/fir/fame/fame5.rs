use petgraph::Direction::Incoming;
use rusty_firrtl::{Annotations, Identifier};
use serde_json::Value;
use indexmap::IndexMap;
use std::collections::HashMap;
use std::collections::HashSet;
use crate::common::graphviz::DefaultGraphVizCore;
use crate::ir::{fir::*, hierarchy::InstPath};


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

fn update_fame5_top(top: &mut FirGraph, host_clock: &Identifier, host_reset: &Identifier, module_to_fame5: &Identifier) {
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
