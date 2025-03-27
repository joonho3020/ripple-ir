use chirrtl_parser::ast::*;
use indexmap::{IndexMap, IndexSet};
use petgraph::graph::NodeIndex;
use crate::ir::firir::*;
use crate::ir::rir::{rgraph::*, rir::*, agg::*, rnode::*, redge::*};

pub fn from_fir(fir: &FirIR) -> RippleIR {
    let mut ret = RippleIR::new(fir.name.clone());

    // Create instance hierarchy
    ret.hierarchy.build_from_fir(fir);

    // Convert FIR graph into Ripple graph
    for hier in ret.hierarchy.topo_order() {
        let fgraph = fir.graphs.get(hier.name()).unwrap();
        let rgraph = from_fir_graph(fgraph);
        ret.graphs.insert(hier.name().clone(), rgraph);
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
        for id in fg.graph.node_indices() {
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
    let mut node_map: IndexMap<NodeIndex, AggNodeIndex> = IndexMap::new();
    let mut ns = NameSpace::new(fg);

    // Create nodes
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
        let name = if let Some(name) = &node.name {
            name.clone()
        } else {
            ns.next()
        };

        let rir_nt = RippleNodeType::from(&node.nt);
        let agg_node = AggNodeData::new(name, rir_nt, node.ttree.clone());
        let agg_id = rg.add_node_agg(agg_node.clone());
        node_map.insert(id, agg_id);
    }

    // Add edges
    for id in fg.graph.edge_indices() {
        let edge = fg.graph.edge_weight(id).unwrap();
        let ep = fg.graph.edge_endpoints(id).unwrap();
        let src = ep.0;
        let dst = ep.1;

        let src_id = *node_map.get(&src).unwrap();
        let src_ref = match &edge.src {
            Expr::Reference(src_ref) => src_ref,
            _ => {
                let agg_src_node = rg.node_weight_agg(src_id).unwrap();
                &Reference::Ref(agg_src_node.name.clone())
            }
        };

        let dst_id = *node_map.get(&dst).unwrap();
        let dst_ref = match &edge.dst {
            Some(dst_ref) => dst_ref,
            None => {
                let agg_dst_node = rg.node_weight_agg(dst_id).unwrap();
                &Reference::Ref(agg_dst_node.name.clone())
            }
        };

        let agg_edge = AggEdgeData::new(RippleEdgeType::from(&edge.et));

        match &edge.et {
            FirEdgeType::MuxCond         |
                FirEdgeType::Clock       |
                FirEdgeType::Reset       |
                FirEdgeType::DontCare    |
                FirEdgeType::PhiSel      |
                FirEdgeType::MemPortAddr |
                FirEdgeType::ArrayAddr   => {
                    rg.add_fanout_edge_agg(src_id, &src_ref, dst_id, dst_ref, agg_edge);
            }
            FirEdgeType::MemPortEdge => {
                rg.add_mem_edge_agg(src_id, &src_ref, dst_id, dst_ref, agg_edge);
            }
            _ => {
                rg.add_edge_agg(src_id, &src_ref, dst_id, dst_ref, agg_edge);
            }
        }
    }
    return rg;
}

#[cfg(test)]
mod test {
    use crate::common::RippleIRErr;
    use crate::passes::runner::run_passes_from_filepath;
    use crate::common::graphviz::GraphViz;
    use super::*;

    fn run_simple(input: &str) -> Result<(), RippleIRErr> {
        let fir = run_passes_from_filepath(input)?;
        for (module, fg) in fir.graphs.iter() {
            fg.export_graphviz(&format!("./test-outputs/{}-{}.fir.pdf",
                    fir.name.to_string(), module.to_string()), None, None, false)?;
        }

        let rir = from_fir(&fir);
        for (module, rg) in rir.graphs.iter() {
            rg.export_graphviz(&format!("./test-outputs/{}-{}.rir.pdf",
                    rir.name.to_string(), module.to_string()), None, None, false)?;
        }
        Ok(())
    }

    #[test]
    fn gcd() {
        run_simple("./test-inputs/GCD.fir")
            .expect("gcd");
    }

    #[test]
    fn decoupledmux() {
        run_simple("./test-inputs/DecoupledMux.fir")
            .expect("decoupledmux");
    }

    #[test]
    fn hierarchy() {
        run_simple("./test-inputs/Hierarchy.fir")
            .expect("hierarchy");
    }

    #[test]
    fn dynamicindexing() {
        run_simple("./test-inputs/DynamicIndexing.fir")
            .expect("dynamicindexing");
    }

    #[test]
    fn singleportsram() {
        run_simple("./test-inputs/SinglePortSRAM.fir")
            .expect("singleportsram");
    }
}
