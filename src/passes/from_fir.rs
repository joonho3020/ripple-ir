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
    let mut node_map: IndexMap<NodeIndex, RootTypeTreeKey> = IndexMap::new();
    let mut ns = NameSpace::new(fg);

    // Create nodes
    for id in fg.node_indices() {
        let node = fg.node_weight(id).unwrap();
        let name = if let Some(name) = &node.name {
            name.clone()
        } else {
            ns.next()
        };

        let root_key = rg.add_aggregate_node(
            name,
            &node.ttree.as_ref().unwrap(),
            RippleNodeType::from(&node.nt));

        node_map.insert(id, root_key);
    }

    // Add edges
    for id in fg.edge_indices() {
        let edge = fg.edge_weight(id).unwrap();
        let ep = fg.edge_endpoints(id).unwrap();
        let src = ep.0;
        let dst = ep.1;

        let src_key = node_map.get(&src).unwrap();
        let dst_key = node_map.get(&dst).unwrap();

        println!("============ src_key {:?} dst_key {:?}", src_key, dst_key);

        match (&edge.src, &edge.dst) {
            (Expr::Reference(src_ref), Some(dst_ref)) => {
                rg.add_aggregate_edge(
                    src_key,
                    src_ref,
                    dst_key,
                    dst_ref,
                    RippleEdgeType::from(&edge.et));
            }
            (Expr::Reference(src_ref), None) => {
                rg.add_aggregate_edge(
                    src_key,
                    src_ref,
                    dst_key,
                    &Reference::Ref(dst_key.name.clone()),
                    RippleEdgeType::from(&edge.et));
            }
            _ => {
                rg.add_aggregate_edge(
                    src_key,
                    &Reference::Ref(src_key.name.clone()),
                    dst_key,
                    &Reference::Ref(dst_key.name.clone()),
                    RippleEdgeType::from(&edge.et));
            }
        }

        println!("=====================================");
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
            fir.graphs.get(module).unwrap()
                .export_graphviz(&format!("./test-outputs/{}-{}.fir.pdf",
                        fir.name.to_string(), module.to_string()), None, true)?;

            rg.export_graphviz(&format!("./test-outputs/{}-{}.rir.pdf",
                    rir.name.to_string(), module.to_string()), None, true)?;
        }
        Ok(())
    }

    #[test]
    fn gcd() {
        run("./test-inputs/GCD.fir")
            .expect("gcd ast assumption");
    }

    #[test]
    fn decoupledmux() {
        run("./test-inputs/DecoupledMux.fir")
            .expect("decoupledmux ast assumption");
    }

    // TODO: add tests for cases where
    // - Mux, primops
    // - There are references to expressions as array addresses
    // - There are instance hierarchies
    // - SRAMs
    //
    // - Write pass that checks that the Phi nodes are all connected to their child node after
    // from_ast
}
