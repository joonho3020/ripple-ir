use crate::ir::hierarchy::Hierarchy;
use crate::passes::fir::fame::*;
use crate::passes::fir::fame::fame5::top::*;
use crate::ir::fir::*;
use multithread::multithread_module;
use rusty_firrtl::{Expr, PrimOp2Expr, Reference, Width};
use rusty_firrtl::Int;
use rusty_firrtl::Identifier;
use indexmap::IndexMap;
use indexmap::IndexSet;
use petgraph::graph::NodeIndex;
use petgraph::Direction::Incoming;
use std::collections::HashSet;

mod top;
mod multithread;

pub type InstModuleMap = IndexMap<Identifier, Identifier>;
pub type ChannelMap = IndexMap<Identifier, Vec<(u32, NodeIndex)>>;
pub type ThreadIdxEqMap = IndexMap<u32, NodeIndex>;
pub type TargetInstPathString = String;

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

    let mut tops: IndexSet<&Identifier> = IndexSet::new();
    for parent in fir.hier.graph.neighbors_directed(mod_to_fame5_hier_id, Incoming) {
        let hier = fir.hier.graph.node_weight(parent).unwrap();
        tops.insert(hier.name());
    }
    assert!(tops.len() == 1, "Module to FAME5 has multiple parent modules");

    let parents = fir.hier.graph.neighbors_directed(mod_to_fame5_hier_id, Incoming);
    let parent_hier_id = parents.into_iter().next().unwrap();
    let parent_module = fir.hier.graph.node_weight(parent_hier_id).unwrap().name();
    let parent_fg = fir.graphs.get_mut(parent_module).unwrap();

    let (top, hostclock) = find_hostclock(&fir.annos);
    assert!(&Identifier::Name(top.module) == parent_module);

    let (top, hostreset) = find_hostreset(&fir.annos);
    assert!(&Identifier::Name(top.module) == parent_module);

    let nthreads = update_fame5_top(parent_fg, &hostclock, &hostreset, mod_to_fame5);

    let modules_to_fame5: HashSet<Identifier> = fir.hier.all_childs(mod_to_fame5)
        .iter()
        .filter(|hid| {
            let name = fir.hier.graph.node_weight(**hid).unwrap().name();
            !fir.graphs.get(name).unwrap().blackbox
        })
        .map(|id| {
            fir.hier.graph.node_weight(*id).unwrap().name().clone()
        })
        .collect();

    let blackboxes: HashSet<Identifier> = fir.graphs
        .iter()
        .filter(|(_name, fg)| {
            fg.blackbox
        }).map(|(name, _fg)| name.clone())
        .collect();

    for mod_name in modules_to_fame5.iter() {
        let mod_to_fame5_graph = fir.graphs.get(mod_name).unwrap();
        let fame5_module = multithread_module(mod_to_fame5_graph, nthreads, &hostclock, &hostreset, &blackboxes);
        fir.graphs.insert(Identifier::Name(fame5_name(mod_name)), fame5_module);
    }
    for mod_name in modules_to_fame5.iter() {
        fir.graphs.swap_remove(mod_name);
    }
    let hier_new = Hierarchy::new(fir);
    fir.hier = hier_new;
}

pub fn add_thread_idx_update(
    fg: &mut FirGraph,
    thread_idx_id: NodeIndex,
    nthreads: u32,
    thread_idx_bits: u32
) {
    // Create comparison node to check if thread_idx == nthreads - 1
    let thread_idx_ref = Expr::Reference(Reference::Ref(
        fg.graph
            .node_weight(thread_idx_id)
            .unwrap()
            .name
            .as_ref()
            .unwrap()
            .clone(),
    ));

    let (max_idx_const_id, max_idx_expr) =
        fg.add_uint_literal((nthreads - 1).into(), thread_idx_bits);
    let (eq_id, eq_expr) = fg.add_primop2(
        PrimOp2Expr::Eq,
        thread_idx_id,
        thread_idx_ref.clone(),
        max_idx_const_id,
        max_idx_expr,
    );

    // Create add node for thread_idx + 1
    let (one_const_id, one_expr) = fg.add_uint_literal(1, thread_idx_bits);
    let (add_id, add_expr) = fg.add_primop2(
        PrimOp2Expr::Add,
        thread_idx_id,
        thread_idx_ref.clone(),
        one_const_id,
        one_expr,
    );

    // Create mux to select between thread_idx + 1 and 0
    let (zero_const_id, zero_expr) = fg.add_uint_literal(0, thread_idx_bits);
    let (mux_id, mux_expr) = fg.add_mux(
        eq_id,
        eq_expr,
        zero_const_id,
        zero_expr,
        add_id,
        add_expr,
    );

    // Connect mux output to thread_idx register's next value
    let thread_idx_name = fg.graph.node_weight(thread_idx_id).unwrap().name.as_ref().unwrap().clone();
    fg.add_wire(
        mux_id,
        mux_expr,
        thread_idx_id,
        Some(Reference::Ref(thread_idx_name)),
    );
}

pub fn add_thread_idx_reg(
    fg: &mut FirGraph,
    nthreads: u32,
    host_clock: &Identifier,
    host_reset: &Identifier
) -> NodeIndex {
    let thread_idx_bits = log2_ceil(nthreads);
    let thread_idx_name = Identifier::Name("threadIdx".to_string());
    let thread_idx_id = fg.graph.add_node(FirNode::new(
            Some(thread_idx_name.clone()),
            FirNodeType::RegReset,
            Some(uint_ttree(thread_idx_bits))));

    let thread_idx_init = FirNode::new(
        None,
        FirNodeType::UIntLiteral(Width(thread_idx_bits), Int::from(0)),
        None);

    let edge = FirEdge::new(
        Expr::UIntInit(Width(thread_idx_bits), Int::from(0)),
        None,
        FirEdgeType::InitValue);

    let thread_idx_init_id = fg.graph.add_node(thread_idx_init);
    fg.graph.add_edge(thread_idx_init_id, thread_idx_id, edge);

    let host_clock_id = find_host_clock_or_reset_id(fg, host_clock);
    let host_reset_id = find_host_clock_or_reset_id(fg, host_reset);

    let hc_edge = FirEdge::new(
        Expr::Reference(Reference::Ref(host_clock.clone())),
        None,
        FirEdgeType::Clock);
    fg.graph.add_edge(host_clock_id, thread_idx_id, hc_edge);

    let hr_edge = FirEdge::new(
        Expr::Reference(Reference::Ref(host_reset.clone())),
        None,
        FirEdgeType::Reset);
    fg.graph.add_edge(host_reset_id, thread_idx_id, hr_edge);

    thread_idx_id
}

/// Find the node ID for host clock or reset by name
pub fn find_host_clock_or_reset_id(fg: &FirGraph, host_node: &Identifier) -> NodeIndex {
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::{common::{read_annos, RippleIRErr}, passes::{ast::{firrtl3_print::FIRRTL3Printer, firrtl3_split_exprs::firrtl3_split_exprs}, fir::{remove_unnecessary_phi::remove_all_phi, to_ast_firrtl3::to_ast_firrtl3}, runner::run_fir_passes_from_circuit}};
    use test_case::test_case;
    use firrtl3_parser::parse_circuit as parse_firrtl3;
    use crate::passes::ast::print::Printer;

    #[test_case("FireSimGCD"; "FireSimGCD")]
    #[test_case("FireSimNestedModels"; "FireSimNestedModels")]
    fn fame5(name: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(format!("./test-inputs-firrtl3/{}.fir", name)).expect("to_exist");
        let mut circuit = parse_firrtl3(&source).expect("firrtl parser");
        circuit.annos = read_annos(&format!("./test-inputs-firrtl3/{}.json", name))?;

        firrtl3_split_exprs(&mut circuit);
        let mut ir = run_fir_passes_from_circuit(&circuit)?;
        remove_all_phi(&mut ir);

        fame5_transform(&mut ir);

        let fame5_ast = to_ast_firrtl3(&ir);
        let mut printer = FIRRTL3Printer::new();
        let fame5_firrtl_str = printer.print_circuit(&fame5_ast);
        let out_path = format!("./test-outputs/{}.{}.firrtl3.fame5.fir", name, circuit.name);
        std::fs::write(&out_path, fame5_firrtl_str)?;

        Ok(())
    }
}
