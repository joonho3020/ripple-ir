use chirrtl_parser::ast::*;
use firir::FirGraph;
use indexmap::{IndexMap, IndexSet};
use petgraph::graph::NodeIndex;
use crate::ir::firir::{FirEdgeType, FirIR};
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
    let mut node_map: IndexMap<NodeIndex, RootTypeTreeKey> = IndexMap::new();
    let mut ns = NameSpace::new(fg);

    // Create nodes
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
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
    for id in fg.graph.edge_indices() {
        let edge = fg.graph.edge_weight(id).unwrap();
        let ep = fg.graph.edge_endpoints(id).unwrap();
        let src = ep.0;
        let dst = ep.1;

        let src_key = node_map.get(&src).unwrap();
        let dst_key = node_map.get(&dst).unwrap();

        let src_ref = match &edge.src {
            Expr::Reference(src_ref) => src_ref,
            _ => &Reference::Ref(src_key.name.clone())
        };

        let dst_ref = match &edge.dst {
            Some(dst_ref) => dst_ref,
            None => &Reference::Ref(dst_key.name.clone()),
        };
        let et = RippleEdgeType::from(&edge.et);

        match &edge.et {
            FirEdgeType::MuxCond     |
                FirEdgeType::Clock       |
                FirEdgeType::Reset       |
                FirEdgeType::DontCare    |
                FirEdgeType::PhiSel      |
                FirEdgeType::MemPortAddr => {
                    rg.add_single_edge(src_key, &src_ref, dst_key, dst_ref, et);
            }
            FirEdgeType::MemPortEdge => {
                rg.add_aggregate_mem_edge(src_key, &src_ref, dst_key, dst_ref, et);
            }
            _ => {
                rg.add_aggregate_edge(src_key, &src_ref, dst_key, dst_ref, et);
            }
        }
    }

    return rg;
}

#[cfg(test)]
mod test {
    use std::collections::VecDeque;
    use graphviz_rust::attributes::{NodeAttributes, color_name};

    use crate::common::graphviz::*;
    use crate::common::RippleIRErr;
    use crate::passes::runner::run_passes_from_filepath;
    use crate::common::graphviz::GraphViz;
    use super::*;

    fn run(input: &str) -> Result<(), RippleIRErr> {
        let fir = run_passes_from_filepath(input)?;
        let rir = from_fir(&fir);
        for (module, rg) in rir.graphs.iter() {
            fir.graphs.get(module).unwrap()
                .export_graphviz(&format!("./test-outputs/{}-{}.fir.pdf",
                        fir.name.to_string(), module.to_string()), None, None, true)?;

            rg.export_graphviz(&format!("./test-outputs/{}-{}.rir.pdf",
                    rir.name.to_string(), module.to_string()), None, None, true)?;
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

    fn traverse(module: &Identifier, rg: &RippleGraph, export: bool) -> Result<u32, RippleIRErr> {
        let mut q: VecDeque<TreeIdx> = VecDeque::new();

        for agg_id in rg.node_indices_agg().iter() {
            let agg_w = rg.node_weight_agg(*agg_id);

            // Collect all the IOs
            match agg_w.1.unwrap().nt {
                RippleNodeType::Input |
                    RippleNodeType::Output => {
                    q.push_back(*agg_id);
                }
                _ => {
                }
            }
        }

        let mut file_names: Vec<String> = vec![];

        let mut iter_outer = 0;
        let mut agg_node_cnt = 0;
        let mut vis_map = rg.vismap_agg();

        while !q.is_empty() {
            let agg_id = q.pop_front().unwrap();
            if vis_map.is_visited(agg_id) {
                continue;
            }
            vis_map.visit(agg_id);
            agg_node_cnt += 1;

            let mut node_attributes = NodeAttributeMap::default();
            let src_ttree = rg.ttrees.get(agg_id as usize).unwrap();
            let leaf_ids = src_ttree.all_leaves();
            for leaf_id in leaf_ids {
                let leaf = src_ttree.graph.node_weight(leaf_id).unwrap();
                let rg_id = leaf.id.unwrap();
                node_attributes.insert(rg_id, NodeAttributes::color(color_name::green));
            }

            let agg_edges = rg.edges_agg(agg_id);
            for (iter_inner, (edge_key, edges)) in agg_edges.iter().enumerate() {
                if !vis_map.is_visited(edge_key.dst_tree) {
                    q.push_back(edge_key.dst_tree);

                    let mut cur_node_attributes = node_attributes.clone();

                    let dst_ttree = rg.ttrees.get(edge_key.dst_tree as usize).unwrap();
                    let leaf_ids = dst_ttree.all_leaves();
                    for leaf_id in leaf_ids {
                        let leaf = dst_ttree.graph.node_weight(leaf_id).unwrap();
                        let rg_id = leaf.id.unwrap();
                        cur_node_attributes.insert(rg_id, NodeAttributes::color(color_name::blue));
                    }

                    let mut edge_attributes = EdgeAttributeMap::default();
                    for eid in edges {
                        edge_attributes.insert(*eid, NodeAttributes::color(color_name::red));
                    }

                    if export {
                        let out_name = format!(
                            "./test-outputs/{}-{}-{}.traverse.agg.pdf",
                            module, iter_outer, iter_inner);

                        rg.export_graphviz(
                            &out_name,
                            Some(cur_node_attributes).as_ref(),
                            Some(edge_attributes).as_ref(),
                            false)?;
                        file_names.push(out_name);
                    }
                }
            }
            iter_outer += 1;
        }
        assert!(!vis_map.has_unvisited());

        if export {
            rg.create_gif(
                &format!("./test-outputs/{}.gif", module),
                &file_names)?;
        }

        Ok(agg_node_cnt)
    }

    fn run_traverse(input: &str) -> Result<(), RippleIRErr> {
        let fir = run_passes_from_filepath(input)?;
        let rir = from_fir(&fir);
        for (module, rg) in rir.graphs.iter() {
            let agg_nodes = traverse(module, rg, false)?;
            let fg = fir.graphs.get(module).unwrap();
            assert_eq!(agg_nodes, fg.graph.node_count() as u32,
                "Node count mismatch expect {} got {}",
                fg.graph.node_count(), agg_nodes);
        }
        Ok(())
    }

    #[test]
    fn traverse_gcd() {
        run_traverse("./test-inputs/GCD.fir")
            .expect("gcd traverse assumption");
    }

    #[test]
    fn traverse_decoupledmux() {
        run_traverse("./test-inputs/DecoupledMux.fir")
            .expect("decoupledmux traverse assumption");
    }

    #[test]
    fn traverse_singleportsram() {
        run_traverse("./test-inputs/SinglePortSRAM.fir")
            .expect("singleportsram traverse assumption");
    }

    #[test]
    fn traverse_aggregatesram() {
        run_traverse("./test-inputs/AggregateSRAM.fir")
            .expect("aggregatesram traverse assumption");
    }


    // TODO: add tests for cases where
    // - There are references to expressions as array addresses
    // - There are instance hierarchies
}
