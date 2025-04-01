use chirrtl_parser::ast::*;

use crate::{common::graphviz::DefaultGraphVizCore, ir::fir::{FirGraph, FirIR, FirNodeType}};

pub fn to_ast(fir: &FirIR) -> Option<Circuit> {
    for (name, fgraph) in fir.graphs.iter() {
    }
    None
}

fn to_circuitmodule(name: &Identifier, fg: &FirGraph) -> CircuitModule {
    if fg.blackbox {
        CircuitModule::ExtModule(to_extmodule(name, fg))
    } else {
        CircuitModule::Module(to_module(name, fg))
    }
}

fn to_extmodule(name: &Identifier, fg: &FirGraph) -> ExtModule {
    let ext_info = &fg.ext_info.as_ref().unwrap();
    let defname = &ext_info.defname;
    let params = &ext_info.params;
    let ports = get_ports(fg);
    ExtModule::new(name.clone(), ports, defname.clone(), params.clone(), Info::default())
}

fn get_ports(fg: &FirGraph) -> Ports {
    let mut ret = Ports::new();
    for id in fg.node_indices() {
        let node = fg.node_weight(id).unwrap();
        match node.nt {
            FirNodeType::Input => {
                ret.push(Box::new(Port::Input(
                    node.name.as_ref().unwrap().clone(),
                    node.ttree.as_ref().unwrap().to_type(),
                    Info::default())));
            }
            FirNodeType::Output => {
                ret.push(Box::new(Port::Output(
                    node.name.as_ref().unwrap().clone(),
                    node.ttree.as_ref().unwrap().to_type(),
                    Info::default())));
            }
            _ => {
                continue;
            }
        }
    }
    return ret;
}


fn to_module(name: &Identifier, fg: &FirGraph) -> Module {
    let ports = get_ports(fg);
    let mut stmts: Stmts = Stmts::new();
    Module::new(name.clone(), ports, stmts, Info::default())
}


#[cfg(test)]
mod test {
}
