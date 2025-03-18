use chirrtl_parser::ast::*;
use firir::FirGraph;
use indexmap::{IndexMap, IndexSet};
use petgraph::graph::NodeIndex;
use crate::common::graphviz::GraphViz;
use crate::ir::firir::FirIR;
use crate::ir::*;


pub fn from_fir(fir: &FirIR) -> RippleIR {
    let mut ret = RippleIR::new(fir.name.clone());
    for (name, fgraph) in fir.graphs.iter() {
        let rg = from_fir_graph(fgraph);
        ret.graphs.insert(name.clone(), rg);
    }
    return ret;
}

// TODO: NameSpace should be part of the IR
struct NameSpace {
    used: IndexSet<Identifier>,
    cntr: u32,
    pfx: String
}

impl NameSpace {
    pub fn new(fg: &FirGraph) -> Self {
        let mut used: IndexSet<Identifier> = IndexSet::new();
        for id in fg.node_indices() {
            let node = fg.graph.node_weight(id).unwrap();
            if let Some(name) = &node.name {
                used.insert(name.clone());
            }
        }
        Self {
            used,
            cntr: 0,
            pfx: "_TMP".to_string(),
        }
    }

    pub fn next(&mut self) -> Identifier {
        loop {
            let candidate = Identifier::Name(format!("{}_{}", self.pfx, self.cntr));
            self.cntr += 1;
            if !self.used.contains(&candidate) {
                return candidate;
            }
        }
    }
}

fn from_fir_graph(fg: &FirGraph) -> RippleGraph {
    let mut rg = RippleGraph::new();
    let mut name_map: IndexMap<NodeIndex, Identifier> = IndexMap::new();
    let mut ns = NameSpace::new(fg);

    // Create name map
    for id in fg.node_indices() {
        let node = fg.node_weight(id).unwrap();
        if let Some(name) = &node.name {
            name_map.insert(id, name.clone());
        } else {
            name_map.insert(id, ns.next());
        }
    }

    for id in fg.node_indices() {
        let node = fg.node_weight(id).unwrap();
        let name = name_map.swap_remove(&id).unwrap();
        rg.add_aggregate_node(
            name,
            &node.ttree.as_ref().unwrap(),
            RippleNodeType::from(&node.nt));
    }

    return rg;
}

#[cfg(test)]
mod test {
    use crate::common::RippleIRErr;
    use crate::passes::runner::run_passes_from_filepath;
    use super::*;

    fn run(input: &str) -> Result<(), RippleIRErr> {
        let fir = run_passes_from_filepath(input)?;
        let rir = from_fir(&fir);
        for (module, rg) in rir.graphs.iter() {
            rg.export_graphviz(&format!("./test-outputs/{}-{}.fir2rir.pdf",
                    rir.name.to_string(), module.to_string()), None, true)?;
        }
        Ok(())
    }

    #[test]
    fn gcd() {
        run("./test-inputs/GCD.fir")
            .expect("gcd ast assumption");
    }
}
