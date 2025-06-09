use rusty_firrtl::{Annotations, Identifier};
use serde_json::Value;
use indexmap::IndexMap;

use crate::ir::{fir::*, hierarchy::InstPath};

pub fn fame5_transform(fir: &mut FirIR) {
    let inst_module_map = fame5_inst_to_module(&filter_fame5_annos(&fir.annos));
    println!("inst_module_map {:?}", inst_module_map);
}

fn filter_fame5_annos(annos: &Annotations) -> Vec<&String> {
    let mut ret = vec![];
    if let Some(annos_list) = annos.0.as_array() {
        for anno in annos_list.iter() {
            if let Some(map) = anno.as_object() {
                if let Some(Value::String(class_value)) = map.get("class") {
                    if class_value == "midas.targetutils.FirrtlEnableModelMultiThreadingAnnotation" {
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

fn fame5_inst_to_module(targets: &Vec<&String>) -> IndexMap<Identifier, Identifier>  {
    let mut inst_to_module: IndexMap<Identifier, Identifier> = IndexMap::new();
    for target in targets {
        let inst_path = InstPath::parse_inst_hierarchy_path(target.as_str());
        let inst_path_leaf = inst_path.last().unwrap();
        inst_to_module.insert(
            Identifier::Name(inst_path_leaf.inst.as_ref().expect("No inst in FAME5 anno").clone()),
            Identifier::Name(inst_path_leaf.module.clone()));
    }
    inst_to_module
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
