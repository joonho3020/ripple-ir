use petgraph::visit::EdgeRef;
use petgraph::Direction::{self, Incoming, Outgoing};
use rusty_firrtl::{Expr, PrimOp2Expr, Reference, Type, TypeAggregate, Width};
use rusty_firrtl::{Annotations, Identifier};
use rusty_firrtl::Int;
use serde_json::Value;
use indexmap::{IndexMap, IndexSet};
use std::collections::HashMap;
use std::collections::HashSet;
use petgraph::graph::{EdgeIndex, NodeIndex};
use crate::common::graphviz::DefaultGraphVizCore;
use crate::ir::typetree::typetree::TypeTree;
use crate::ir::{fir::*, hierarchy::InstPath};
use crate::passes::fir::fame::log2_ceil;
use crate::passes::fir::fame::uint_ttree;
use crate::ir::typetree::tnode::TypeDirection::Outgoing as TypeDirectionOutgoing;

type InstModuleMap = IndexMap<Identifier, Identifier>;
type ChannelMap = IndexMap<Identifier, Vec<(u32, NodeIndex)>>;

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
    let parent_module = fir.hier.node_weight(parent_hier_id).unwrap().name();
    let parent_fg = fir.graphs.get_mut(parent_module).unwrap();

    let (top, hostclock) = find_hostclock(&fir.annos);
    assert!(&Identifier::Name(top.module) == parent_module);

    let (top, hostreset) = find_hostreset(&fir.annos);
    assert!(&Identifier::Name(top.module) == parent_module);

    let nthreads = update_fame5_top(parent_fg, &hostclock, &hostreset, mod_to_fame5);

// let _ = parent_fg.export_graphviz(
// &format!("./test-outputs/{}.fametop.pdf", fir.name),
// None, None, false);

// let fame5_fg = fir.graphs.get(mod_to_fame5).unwrap();
// let _ = fame5_fg.export_graphviz(
// &format!("./test-outputs/{}.tothread.pdf", fir.name),
// None, None, false);

    let mod_to_fame5_graph = fir.graphs.get(mod_to_fame5).unwrap();
    let fame5_module = multithread_module(mod_to_fame5_graph, nthreads, &hostclock, &hostreset);

    fir.graphs.swap_remove(mod_to_fame5);
    fir.add_module(Identifier::Name(fame5_name(mod_to_fame5)), fame5_module);
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

fn add_thread_idx_update(
    fg: &mut FirGraph,
    thread_idx_id: NodeIndex,
    nthreads: u32,
    thread_idx_bits: u32
) {
    // Create comparison node to check if thread_idx == nthreads - 1
    let eq_name = fg.namespace.next();
    let eq_node = FirNode::new(
        Some(eq_name.clone()),
        FirNodeType::PrimOp2Expr(PrimOp2Expr::Eq),
        None
    );
    let eq_id = fg.graph.add_node(eq_node);

    // Connect thread_idx to eq node
    let thread_idx_name = fg.graph.node_weight(thread_idx_id).unwrap().name.as_ref().unwrap().clone();
    let thread_idx_ref = Expr::Reference(Reference::Ref(thread_idx_name.clone()));
    let op0 = FirEdge::new(thread_idx_ref.clone(), None, FirEdgeType::Operand0);
    fg.graph.add_edge(thread_idx_id, eq_id, op0);

    // Connect nthreads - 1 constant to eq node
    let max_idx_const = FirNode::new(
        None,
        FirNodeType::UIntLiteral(Width(thread_idx_bits), Int::from(nthreads - 1)),
        None);
    let max_idx_const_id = fg.graph.add_node(max_idx_const);
    let op1 = FirEdge::new(
        Expr::UIntInit(Width(thread_idx_bits), Int::from(nthreads - 1)),
        None,
        FirEdgeType::Operand1);
    fg.graph.add_edge(max_idx_const_id, eq_id, op1);

    // Create add node for thread_idx + 1
    let add_name = fg.namespace.next();
    let add_node = FirNode::new(
        Some(add_name.clone()),
        FirNodeType::PrimOp2Expr(PrimOp2Expr::Add),
        None
    );
    let add_id = fg.graph.add_node(add_node);

    // Connect thread_idx to add node
    let add_op0 = FirEdge::new(thread_idx_ref.clone(), None, FirEdgeType::Operand0);
    fg.graph.add_edge(thread_idx_id, add_id, add_op0);

    // Connect constant 1 to add node
    let one_const = FirNode::new(
        None,
        FirNodeType::UIntLiteral(Width(thread_idx_bits), Int::from(1)),
        None);
    let one_const_id = fg.graph.add_node(one_const);
    let add_op1 = FirEdge::new(
        Expr::UIntInit(Width(thread_idx_bits), Int::from(1)),
        None,
        FirEdgeType::Operand1);
    fg.graph.add_edge(one_const_id, add_id, add_op1);

    // Create mux to select between thread_idx + 1 and 0
    let mux_name = fg.namespace.next();
    let mux_node = FirNode::new(
        Some(mux_name.clone()),
        FirNodeType::Mux,
        None
    );
    let mux_id = fg.graph.add_node(mux_node);

    // Connect eq result to mux condition
    let cond = FirEdge::new(
        Expr::Reference(Reference::Ref(eq_name)),
        None,
        FirEdgeType::MuxCond
    );
    fg.graph.add_edge(eq_id, mux_id, cond);

    // Connect thread_idx + 1 to mux true input
    let true_in = FirEdge::new(
        Expr::Reference(Reference::Ref(add_name)),
        None,
        FirEdgeType::MuxFalse
    );
    fg.graph.add_edge(add_id, mux_id, true_in);

    // Connect constant 0 to mux false input
    let zero_const = FirNode::new(
        None,
        FirNodeType::UIntLiteral(Width(thread_idx_bits), Int::from(0)),
        None);
    let zero_const_id = fg.graph.add_node(zero_const);
    let false_in = FirEdge::new(
        Expr::UIntInit(Width(thread_idx_bits), Int::from(0)),
        None,
        FirEdgeType::MuxTrue
    );
    fg.graph.add_edge(zero_const_id, mux_id, false_in);

    // Connect mux output to thread_idx register's next value
    let next_edge = FirEdge::new(
        Expr::Reference(Reference::Ref(mux_name)),
        Some(Reference::Ref(thread_idx_name.clone())),
        FirEdgeType::Wire
    );
    fg.graph.add_edge(mux_id, thread_idx_id, next_edge);
}

fn fame5_name(name: &Identifier) -> String {
    format!("{}_fame5", name)
}

fn update_fame5_top(
    top: &mut FirGraph,
    host_clock: &Identifier,
    host_reset: &Identifier,
    module_to_fame5: &Identifier
) -> u32 {
    let mut inst_ids: Vec<NodeIndex> = top.graph.node_indices().into_iter()
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

    let nthreads = inst_ids.len() as u32;
    let thread_idx_id = add_thread_idx_reg(top, nthreads, host_clock, host_reset);
    add_thread_idx_update(top, thread_idx_id, nthreads, log2_ceil(nthreads + 1));

    let fame5_inst_name = Identifier::Name(fame5_name(module_to_fame5));
    let fame5_mod_name  = Identifier::Name(fame5_name(module_to_fame5));
    assert!(!top.namespace.contains(&fame5_inst_name));

    let fame5_inst_id = top.graph.add_node(
        FirNode::new(
            Some(fame5_inst_name),
            FirNodeType::Inst(fame5_mod_name),
            None));
    connect_to_host_clock_and_reset(top, host_clock, host_reset, fame5_inst_id);

    let mut input_channel_map: ChannelMap = ChannelMap::new();
    let mut output_channel_map: ChannelMap = ChannelMap::new();
    for (idx, &inst_id) in inst_ids.iter().enumerate() {
        for eid in top.graph.edges_directed(inst_id, Incoming) {
            let edge = top.graph.edge_weight(eid.id()).unwrap();
            let ep = top.graph.edge_endpoints(eid.id()).unwrap();
            let src_node = top.graph.node_weight(ep.0).unwrap();

            let dst_ref = edge.dst.as_ref().expect("Channel destination reference");
            match dst_ref {
                Reference::RefDot(_, channel_name) => {
                    if channel_name == host_clock || channel_name == host_reset {
                        continue;
                    } else if src_node.nt == FirNodeType::Input {
                        if !input_channel_map.contains_key(channel_name) {
                            input_channel_map.insert(channel_name.clone(), vec![]);
                        }
                        input_channel_map.get_mut(channel_name).unwrap().push((idx as u32, ep.0));
                    } else {
                        if !output_channel_map.contains_key(channel_name) {
                            output_channel_map.insert(channel_name.clone(), vec![]);
                        }
                        output_channel_map.get_mut(channel_name).unwrap().push((idx as u32, ep.0));
                    }
                }
                _ => {
                    unreachable!();
                }
            }
        }
    }

    // HACK: Flip the output channels as the edge directionality is currently flipped in FIR
    let mut edges_to_flip: Vec<EdgeIndex> = vec![];
    for id in top.graph.node_indices() {
        let node = top.graph.node_weight(id).unwrap();
        if node.nt == FirNodeType::Output {
            for eid in top.graph.edges_directed(id, Outgoing) {
                edges_to_flip.push(eid.id());
            }
        }
    }

    for &eid in edges_to_flip.iter() {
        let w = top.graph.edge_weight(eid).unwrap().clone();
        let dst_ref  = w.dst.unwrap();
        if let Expr::Reference(src_ref) = w.src {
            let edge = FirEdge::new(Expr::Reference(dst_ref), Some(src_ref), w.et);
            let ep = top.graph.edge_endpoints(eid).unwrap();
            top.graph.add_edge(ep.1, ep.0, edge);
        }
    }

    edges_to_flip.sort();
    for &eid in edges_to_flip.iter().rev() {
        top.graph.remove_edge(eid);
    }

    connect_libdn_input_signals(
        top,
        &input_channel_map,
        thread_idx_id,
        fame5_inst_id,
        log2_ceil(nthreads + 1),
        Identifier::Name("valid".to_string()));

    connect_libdn_input_signals(
        top,
        &input_channel_map,
        thread_idx_id,
        fame5_inst_id,
        log2_ceil(nthreads + 1),
        Identifier::Name("bits".to_string()));

    connect_libdn_output_signals(
        top,
        &input_channel_map,
        thread_idx_id,
        fame5_inst_id,
        log2_ceil(nthreads + 1),
        Identifier::Name("ready".to_string()));

    connect_libdn_input_signals(
        top,
        &output_channel_map,
        thread_idx_id,
        fame5_inst_id,
        log2_ceil(nthreads + 1),
        Identifier::Name("ready".to_string()));

    connect_libdn_output_signals(
        top,
        &output_channel_map,
        thread_idx_id,
        fame5_inst_id,
        log2_ceil(nthreads + 1),
        Identifier::Name("valid".to_string()));

    broadcase_libdn_output_signals(
        top,
        &output_channel_map,
        fame5_inst_id,
        Identifier::Name("bits".to_string()));

    inst_ids.sort();
    for id in inst_ids.iter().rev() {
        top.graph.remove_node(*id);
    }

    nthreads
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

fn add_thread_idx_reg(
    fg: &mut FirGraph,
    nthreads: u32,
    host_clock: &Identifier,
    host_reset: &Identifier
) -> NodeIndex {
    let thread_idx_bits = log2_ceil(nthreads + 1);
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
        Some(Reference::RefDot(Box::new(Reference::Ref(name.clone())), host_clock.clone())),
        FirEdgeType::Clock);
    fg.graph.add_edge(host_clock_id, dst_id, hc_edge);

    let hr_edge = FirEdge::new(
        Expr::Reference(Reference::Ref(host_reset.clone())),
        Some(Reference::RefDot(Box::new(Reference::Ref(name.clone())), host_reset.clone())),
        FirEdgeType::Reset);
    fg.graph.add_edge(host_reset_id, dst_id, hr_edge);
}

fn create_thread_id_select_mux(
    fg: &mut FirGraph,
    cur_thread_idx: u32,
    thread_idx_id: NodeIndex,
    thread_idx_bits: u32,
    channel_id: NodeIndex,
    signal_name: Identifier
) -> NodeIndex {
    let node = fg.graph.node_weight(channel_id).unwrap();
    let signal_ref = Expr::Reference(Reference::RefDot(
            Box::new(Reference::Ref(node.name.as_ref().unwrap().clone())),
            signal_name
    ));

    // Create eq node to compare thread_idx with current index
    let eq_name = fg.namespace.next();
    let eq_node = FirNode::new(
        Some(eq_name.clone()),
        FirNodeType::PrimOp2Expr(PrimOp2Expr::Eq),
        None
    );
    let eq_id = fg.graph.add_node(eq_node);

    // Connect thread_idx to eq node
    let thread_idx_ref = Expr::Reference(Reference::Ref(Identifier::Name("threadIdx".to_string())));
    let op0 = FirEdge::new(thread_idx_ref, None, FirEdgeType::Operand0);
    fg.graph.add_edge(thread_idx_id, eq_id, op0);

    // Connect constant index to eq node
    let idx_const = FirNode::new(
        None,
        FirNodeType::UIntLiteral(Width(thread_idx_bits), Int::from(cur_thread_idx)),
        None);
    let idx_const_id = fg.graph.add_node(idx_const);
    let op1 = FirEdge::new(
        Expr::UIntInit(Width(thread_idx_bits), Int::from(cur_thread_idx)),
        None,
        FirEdgeType::Operand1);
    fg.graph.add_edge(idx_const_id, eq_id, op1);

    // Create mux node
    let mux_name = fg.namespace.next();
    let mux_node = FirNode::new(
        Some(mux_name.clone()),
        FirNodeType::Mux,
        None
    );
    let mux_id = fg.graph.add_node(mux_node);

    // Connect eq result to mux condition
    let cond = FirEdge::new(
        Expr::Reference(Reference::Ref(eq_name)),
        None,
        FirEdgeType::MuxCond
    );
    fg.graph.add_edge(eq_id, mux_id, cond);

    // Connect input signal to mux true input
    let true_in = FirEdge::new(signal_ref, None, FirEdgeType::MuxTrue);
    fg.graph.add_edge(channel_id, mux_id, true_in);

    mux_id
}

fn connect_libdn_input_signals(
    fg: &mut FirGraph,
    input_channel_map: &ChannelMap,
    thread_idx_id: NodeIndex,
    fame5_inst_id: NodeIndex,
    thread_idx_bits: u32,
    signal_name: Identifier
)
{
    for (channel_name, inputs) in input_channel_map.iter() {
        let mut prev_mux_false_id: Option<NodeIndex> = None;
        let mut prev_mux_false_src: Option<Expr> = None;
        let mut prev_mux_false_et = FirEdgeType::MuxFalse;

        for (cur_idx, cur_channel_id) in inputs.iter() {
            if *cur_idx == 0 {
                let node = fg.graph.node_weight(*cur_channel_id).unwrap();
                let signal_ref = Expr::Reference(Reference::RefDot(
                        Box::new(Reference::Ref(node.name.as_ref().unwrap().clone())),
                        signal_name.clone()
                ));
                prev_mux_false_id = Some(*cur_channel_id);
                prev_mux_false_src = Some(signal_ref);
                continue;
            }

            // Create a mux and connect it with the previous signal
            let mux_id = create_thread_id_select_mux(fg, *cur_idx, thread_idx_id, thread_idx_bits, *cur_channel_id, signal_name.clone());
            let edge = FirEdge::new(prev_mux_false_src.as_ref().unwrap().clone(), None, prev_mux_false_et.clone());
            fg.graph.add_edge(prev_mux_false_id.unwrap(), mux_id, edge);

            // Update prev_mux_false_*
            let mux_name = fg.graph.node_weight(mux_id).unwrap().name.as_ref().unwrap();
            prev_mux_false_id = Some(mux_id);
            prev_mux_false_src = Some(Expr::Reference(Reference::Ref(mux_name.clone())));
            prev_mux_false_et = FirEdgeType::Wire;

            // If last mux, connect it to the threaded instance
            if *cur_idx as usize == inputs.len() - 1 {
                let fame5_inst_name = fg.graph.node_weight(fame5_inst_id).unwrap().name.as_ref().unwrap();
                let dst_channel_ref = Reference::RefDot(Box::new(Reference::Ref(fame5_inst_name.clone())), channel_name.clone());
                let dst_ref = Reference::RefDot(Box::new(dst_channel_ref), signal_name.clone());
                let edge = FirEdge::new(prev_mux_false_src.as_ref().unwrap().clone(), Some(dst_ref), FirEdgeType::Wire);
                fg.graph.add_edge(mux_id, fame5_inst_id, edge);
            }
        }
    }
}

fn broadcase_libdn_output_signals(
    fg: &mut FirGraph,
    output_channel_map: &ChannelMap,
    fame5_inst_id: NodeIndex,
    signal_name: Identifier
)
{
    for (channel_name, outputs) in output_channel_map.iter() {
        for (_cur_idx, cur_channel_id) in outputs.iter() {
            // Get signal from FAME5 instance
            let fame5_inst_name = fg.graph.node_weight(fame5_inst_id).unwrap().name.as_ref().unwrap();
            let fame5_signal_ref = Expr::Reference(Reference::RefDot(
                Box::new(Reference::RefDot(
                    Box::new(Reference::Ref(fame5_inst_name.clone())),
                    channel_name.clone()
                )),
                signal_name.clone()
            ));

            // Connect AND result to ready signal of source node
            let node = fg.graph.node_weight(*cur_channel_id).unwrap();
            let dst = Reference::RefDot(
                Box::new(Reference::Ref(node.name.as_ref().unwrap().clone())),
                signal_name.clone());
            let edge = FirEdge::new(
                fame5_signal_ref,
                Some(dst),
                FirEdgeType::Wire
            );
            fg.graph.add_edge(fame5_inst_id, *cur_channel_id, edge);
        }
    }
}


fn connect_libdn_output_signals(
    fg: &mut FirGraph,
    output_channel_map: &ChannelMap,
    thread_idx_id: NodeIndex,
    fame5_inst_id: NodeIndex,
    thread_idx_bits: u32,
    signal_name: Identifier
)
{
    for (channel_name, outputs) in output_channel_map.iter() {
        for (cur_idx, cur_channel_id) in outputs.iter() {
            // Create eq node to compare thread_idx with current index
            let eq_name = fg.namespace.next();
            let eq_node = FirNode::new(
                Some(eq_name.clone()),
                FirNodeType::PrimOp2Expr(PrimOp2Expr::Eq),
                None
            );
            let eq_id = fg.graph.add_node(eq_node);

            // Connect thread_idx to eq node
            let thread_idx_name = fg.graph.node_weight(thread_idx_id).unwrap().name.as_ref().unwrap();
            let thread_idx_ref = Expr::Reference(Reference::Ref(thread_idx_name.clone()));
            let op0 = FirEdge::new(thread_idx_ref, None, FirEdgeType::Operand0);
            fg.graph.add_edge(thread_idx_id, eq_id, op0);

            // Connect constant index to eq node
            let idx_const = FirNode::new(
                None,
                FirNodeType::UIntLiteral(Width(thread_idx_bits), Int::from(*cur_idx)),
                None);
            let idx_const_id = fg.graph.add_node(idx_const);
            let op1 = FirEdge::new(
                Expr::UIntInit(Width(thread_idx_bits), Int::from(*cur_idx)),
                None,
                FirEdgeType::Operand1);
            fg.graph.add_edge(idx_const_id, eq_id, op1);

            // Get signal from FAME5 instance
            let fame5_inst_name = fg.graph.node_weight(fame5_inst_id).unwrap().name.as_ref().unwrap();
            let fame5_signal_ref = Expr::Reference(Reference::RefDot(
                Box::new(Reference::RefDot(
                    Box::new(Reference::Ref(fame5_inst_name.clone())),
                    channel_name.clone()
                )),
                signal_name.clone()
            ));

            // Create AND node to combine eq result with FAME5 signal
            let and_name = fg.namespace.next();
            let and_node = FirNode::new(
                Some(and_name.clone()),
                FirNodeType::PrimOp2Expr(PrimOp2Expr::And),
                None
            );
            let and_id = fg.graph.add_node(and_node);

            // Connect eq result to AND node
            let eq_edge = FirEdge::new(
                Expr::Reference(Reference::Ref(eq_name)),
                None,
                FirEdgeType::Operand0
            );
            fg.graph.add_edge(eq_id, and_id, eq_edge);

            // Connect FAME5 signal to AND node
            let fame5_edge = FirEdge::new(
                fame5_signal_ref,
                None,
                FirEdgeType::Operand1
            );
            fg.graph.add_edge(fame5_inst_id, and_id, fame5_edge);

            // Connect AND result to ready signal of source node
            let node = fg.graph.node_weight(*cur_channel_id).unwrap();
            let ready_dst = Reference::RefDot(
                Box::new(Reference::Ref(node.name.as_ref().unwrap().clone())),
                signal_name.clone());
            let ready_edge = FirEdge::new(
                Expr::Reference(Reference::Ref(and_name)),
                Some(ready_dst),
                FirEdgeType::Wire
            );
            fg.graph.add_edge(and_id, *cur_channel_id, ready_edge);
        }
    }
}

fn find_edge_with_type(fg: &FirGraph, id: NodeIndex, et: FirEdgeType, dir: Direction) -> Option<EdgeIndex> {
    for eid in fg.graph.edges_directed(id, dir) {
        let edge = fg.graph.edge_weight(eid.id()).unwrap();
        if edge.et == et {
            return Some(eid.id());
        }
    }
    None
}

fn multithread_module(
    fg: &FirGraph,
    nthreads: u32,
    host_clock: &Identifier,
    host_reset: &Identifier
) -> FirGraph {
    let mut fame5 = fg.clone();

    let thread_idx_id = add_thread_idx_reg(&mut fame5, nthreads, host_clock, host_reset);
    let thread_idx_name = fame5.graph.node_weight(thread_idx_id).unwrap().name.as_ref().unwrap().clone();
    add_thread_idx_update(&mut fame5, thread_idx_id, nthreads, log2_ceil(nthreads + 1));

    let mut reg_ids: Vec<NodeIndex> = fg.graph.node_indices().filter(|id| {
        let node = fg.graph.node_weight(*id).unwrap();
        node.nt == FirNodeType::Reg ||
            node.nt == FirNodeType::RegReset
    }).collect();

    reg_ids.sort();
    for &reg_id in reg_ids.iter().rev() {
        fame5.graph.remove_node(reg_id);

        let node = fg.graph.node_weight(reg_id).unwrap();
        let tpe = node.ttree.as_ref().unwrap().to_type();
        let tpe_arr = Type::TypeAggregate(Box::new(
                TypeAggregate::Array(
                    Box::new(tpe),
                    Int::from(nthreads))));

        let reg_name = node.name.as_ref().unwrap();

        // Make register into array
        let reg_threaded = FirNode::new(
            Some(reg_name.clone()),
            FirNodeType::Reg,
            Some(TypeTree::build_from_type(&tpe_arr, TypeDirectionOutgoing)));

        let reg_threaded_id = fame5.graph.add_node(reg_threaded);

        // Add clock to threaded register
        let clock_eid = find_edge_with_type(fg, reg_id, FirEdgeType::Clock, Incoming).unwrap();
        let clock_id = fg.graph.edge_endpoints(clock_eid).unwrap().0;
        let clock_edge = fg.graph.edge_weight(clock_eid).unwrap().clone();
        fame5.graph.add_edge(clock_id, reg_threaded_id, clock_edge);

        // Add reset to threaded register
        if let Some(reset_eid) = find_edge_with_type(fg, reg_id, FirEdgeType::Reset, Incoming) {
            let reset_id = fg.graph.edge_endpoints(reset_eid).unwrap().0;
            let reset_edge = fg.graph.edge_weight(reset_eid).unwrap().clone();
            fame5.graph.add_edge(reset_id, reg_threaded_id, reset_edge);
        }

        // Add register driver edge
        if let Some(driver_eid) = find_edge_with_type(fg, reg_id, FirEdgeType::Wire, Incoming) {
            let driver_id = fg.graph.edge_endpoints(driver_eid).unwrap().0;
            let mut driver_edge = fg.graph.edge_weight(driver_eid).unwrap().clone();
            driver_edge.dst = Some(Reference::RefIdxExpr(
                        Box::new(Reference::Ref(reg_name.clone())),
                        Box::new(Expr::Reference(Reference::Ref(thread_idx_name.clone())))));
            fame5.graph.add_edge(driver_id, reg_threaded_id, driver_edge);
        }

        // Add register sink edges
        for eid in fg.graph.edges_directed(reg_id, Outgoing) {
            let mut edge = fg.graph.edge_weight(eid.id()).unwrap().clone();
            edge.src = Expr::Reference(Reference::RefIdxExpr(
                    Box::new(Reference::Ref(reg_name.clone())),
                    Box::new(Expr::Reference(Reference::Ref(thread_idx_name.clone())))));

            let ep = fg.graph.edge_endpoints(eid.id()).unwrap();
            fame5.graph.add_edge(reg_threaded_id, ep.1, edge);
        }
    }

    // TODO: handle printf and asserts properly
    let mut print_assert_ids: Vec<NodeIndex> = fame5.graph.node_indices().filter(|id| {
        let node = fame5.graph.node_weight(*id).unwrap();
        match node.nt {
            FirNodeType::Printf(..) |
                FirNodeType::Assert(..) => {
                true
            }
            _ => {
                false
            }
        }
    }).collect();

    print_assert_ids.sort();
    for &id in print_assert_ids.iter().rev() {
        fame5.graph.remove_node(id);
    }

    fame5
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{common::{read_annos, RippleIRErr}, passes::{ast::{firrtl3_print::FIRRTL3Printer, firrtl3_split_exprs::firrtl3_split_exprs}, fir::{remove_unnecessary_phi::remove_all_phi, to_ast_firrtl3::to_ast_firrtl3}, runner::run_fir_passes_from_circuit}};
    use test_case::test_case;
    use firrtl3_parser::parse_circuit as parse_firrtl3;
    use crate::passes::ast::print::Printer;

    #[test_case("FireSimGCD"; "FireSimGCD")]
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
        let out_path = format!("./test-outputs/{}.firrtl3.fame5.fir", circuit.name);
        std::fs::write(&out_path, fame5_firrtl_str)?;

        Ok(())
    }
}
