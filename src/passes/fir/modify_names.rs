use chirrtl_parser::ast::Identifier;
use petgraph::Direction::Incoming;

use crate::ir::fir::{FirIR, FirNodeType};

/// Add sfx to all the module names in the module hierarchy
pub fn add_sfx_to_module_names(fir: &mut FirIR, sfx: &str) {
    let old_hier = fir.hier.clone();
    for old in old_hier.topo_order() {
        let mut new_name_str = old.name().to_string();
        new_name_str.push_str(sfx);
        let new_name = Identifier::Name(new_name_str);
        change_name(fir, old.name(), new_name);
    }
}

/// Find a module with name `cur` and change its name to `new`
pub fn change_name(fir: &mut FirIR, cur: &Identifier, new: Identifier) {
    assert!(&new != cur);
    assert!(fir.hier.id(&new).is_none(),
        "Name {:?} already exists in the module hierarchy", new);

    let hid = fir.hier.id(cur).unwrap();

    // Iterate over modules that contain the current module
    for hpid in fir.hier.graph.neighbors_directed(hid, Incoming) {
        let hnode = fir.hier.graph.node_weight(hpid).unwrap();
        let fg = fir.graphs.get_mut(hnode.name()).unwrap();

        // Find modules with cur name and change it to new
        for id in fg.graph.node_indices() {
            let node = fg.graph.node_weight_mut(id).unwrap();
            if let FirNodeType::Inst(module_name) = &mut node.nt {
                if module_name == cur {
                    node.nt = FirNodeType::Inst(new.clone());
                }
            }
        }
    }

    // Update hierarchy tree node
    fir.hier.graph.node_weight_mut(hid).unwrap().set_name(new.clone());

    // Replace graphs with new key
    if let Some(fg) = fir.graphs.swap_remove(cur) {
        fir.graphs.insert(new.clone(), fg);
    }

    if hid == fir.hier.top().unwrap() {
        fir.name = new.clone();
    }
}
