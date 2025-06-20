use crate::passes::fir::fame::*;
use crate::passes::fir::fame::fame5::*;
use crate::ir::hierarchy::InstPath;
use petgraph::Direction::{Incoming, Outgoing};
use petgraph::visit::EdgeRef;
use petgraph::graph::EdgeIndex;
use std::collections::HashMap;
use std::collections::HashSet;
use rusty_firrtl::{Annotations, Identifier};
use rusty_firrtl::{Expr, PrimOp2Expr, Reference};

fn find_class_target_annos<'a>(annos: &'a Annotations, class: &str) -> Vec<&'a TargetInstPathString> {
    let mut ret = vec![];
    if let Some(annos_list) = annos.0.as_array() {
        for anno in annos_list.iter() {
            if let Some(map) = anno.as_object() {
                if let Some(serde_json::Value::String(class_value)) = map.get("class") {
                    if class_value == class {
                        if let Some(serde_json::Value::String(target_value)) = map.get("target") {
                            ret.push(target_value);
                        }
                    }
                }
            }
        }
    }
    ret
}

pub fn find_fame5_target_inst_paths(annos: &Annotations) -> Vec<&TargetInstPathString> {
    find_class_target_annos(annos, "midas.targetutils.FirrtlEnableModelMultiThreadingAnnotation")
}

pub fn get_fame5_inst_to_module_map(targets: &Vec<&TargetInstPathString>) -> InstModuleMap  {
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

pub fn find_hostclock(annos: &Annotations) -> (InstPath, Identifier) {
    find_hostclock_or_reset(annos, "midas.passes.fame.FAMEHostClock")
}

pub fn find_hostreset(annos: &Annotations) -> (InstPath, Identifier) {
    find_hostclock_or_reset(annos, "midas.passes.fame.FAMEHostReset")
}

pub fn module_to_fame5(inst_to_module: &InstModuleMap) -> (Vec<&Identifier>, Option<&Identifier>) {
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

pub fn fame5_name(name: &Identifier) -> String {
    format!("{}_fame5", name)
}

pub fn update_fame5_top(
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
    add_thread_idx_update(top, thread_idx_id, nthreads, log2_ceil(nthreads));

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

    let thread_idx_bits = log2_ceil(nthreads);
    let mut eq_id_map: ThreadIdxEqMap = ThreadIdxEqMap::new();
    for id in 0..nthreads {
        let eq_id = create_thread_id_cmp(top, id, thread_idx_id, thread_idx_bits);
        eq_id_map.insert(id, eq_id);
    }

    connect_libdn_input_signals(
        top,
        &input_channel_map,
        &eq_id_map,
        fame5_inst_id,
        Identifier::Name("valid".to_string()));

    connect_libdn_input_signals(
        top,
        &input_channel_map,
        &eq_id_map,
        fame5_inst_id,
        Identifier::Name("bits".to_string()));

    connect_libdn_output_signals(
        top,
        &input_channel_map,
        &eq_id_map,
        fame5_inst_id,
        Identifier::Name("ready".to_string()));

    connect_libdn_input_signals(
        top,
        &output_channel_map,
        &eq_id_map,
        fame5_inst_id,
        Identifier::Name("ready".to_string()));

    connect_libdn_output_signals(
        top,
        &output_channel_map,
        &eq_id_map,
        fame5_inst_id,
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

fn create_thread_id_cmp(
    fg: &mut FirGraph,
    cur_thread_idx: u32,
    thread_idx_id: NodeIndex,
    thread_idx_bits: u32,
) -> NodeIndex {
    // Create eq node to compare thread_idx with current index
    let thread_idx_ref = Expr::Reference(Reference::Ref(Identifier::Name("threadIdx".to_string())));

    // Connect constant index to eq node
    let (idx_const_id, idx_const_expr) =
        fg.add_uint_literal(cur_thread_idx.into(), thread_idx_bits);

    let (eq_id, _) = fg.add_primop2(
        PrimOp2Expr::Eq,
        thread_idx_id,
        thread_idx_ref,
        idx_const_id,
        idx_const_expr,
    );
    eq_id
}

fn create_thread_id_select_mux(
    fg: &mut FirGraph,
    eq_id: NodeIndex,
    true_id: NodeIndex,
    true_expr: Expr,
    false_id: NodeIndex,
    false_expr: Expr,
) -> (NodeIndex, Expr) {
    let eq_name = fg.graph.node_weight(eq_id).unwrap().name.as_ref().unwrap().clone();
    let eq_expr = Expr::Reference(Reference::Ref(eq_name));
    fg.add_mux(eq_id, eq_expr, true_id, true_expr, false_id, false_expr)
}

fn connect_libdn_input_signals(
    fg: &mut FirGraph,
    input_channel_map: &ChannelMap,
    eq_node_map: &ThreadIdxEqMap,
    fame5_inst_id: NodeIndex,
    signal_name: Identifier
)
{
    for (channel_name, inputs) in input_channel_map.iter() {
        let mut prev_mux_id: Option<NodeIndex> = None;
        let mut prev_mux_expr: Option<Expr> = None;

        for (cur_idx, cur_channel_id) in inputs.iter() {
            let node = fg.graph.node_weight(*cur_channel_id).unwrap();
            let signal_expr = Expr::Reference(Reference::RefDot(
                Box::new(Reference::Ref(node.name.as_ref().unwrap().clone())),
                signal_name.clone(),
            ));

            if *cur_idx == 0 {
                prev_mux_id = Some(*cur_channel_id);
                prev_mux_expr = Some(signal_expr);
                continue;
            }

            let eq_id = eq_node_map.get(cur_idx).unwrap();

            // Create a mux and connect it with the previous signal
            let (mux_id, mux_expr) = create_thread_id_select_mux(
                fg,
                *eq_id,
                *cur_channel_id,
                signal_expr,
                prev_mux_id.unwrap(),
                prev_mux_expr.as_ref().unwrap().clone(),
            );

            // Update prev_mux_false_*
            prev_mux_id = Some(mux_id);
            prev_mux_expr = Some(mux_expr);

            // If last mux, connect it to the threaded instance
            if *cur_idx as usize == inputs.len() - 1 {
                let fame5_inst_name = fg
                    .graph
                    .node_weight(fame5_inst_id)
                    .unwrap()
                    .name
                    .as_ref()
                    .unwrap();
                let dst_channel_ref = Reference::RefDot(
                    Box::new(Reference::Ref(fame5_inst_name.clone())),
                    channel_name.clone(),
                );
                let dst_ref =
                    Reference::RefDot(Box::new(dst_channel_ref), signal_name.clone());
                fg.add_wire(
                    mux_id,
                    prev_mux_expr.as_ref().unwrap().clone(),
                    fame5_inst_id,
                    Some(dst_ref),
                );
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
    eq_node_map: &ThreadIdxEqMap,
    fame5_inst_id: NodeIndex,
    signal_name: Identifier
)
{
    for (channel_name, outputs) in output_channel_map.iter() {
        for (cur_idx, cur_channel_id) in outputs.iter() {
            // Get signal from FAME5 instance
            let fame5_inst_name = fg.graph.node_weight(fame5_inst_id).unwrap().name.as_ref().unwrap();
            let fame5_signal_ref = Expr::Reference(Reference::RefDot(
                Box::new(Reference::RefDot(
                    Box::new(Reference::Ref(fame5_inst_name.clone())),
                    channel_name.clone(),
                )),
                signal_name.clone()
            ));

            // Create AND node to combine eq result with FAME5 signal
            let eq_id = eq_node_map.get(cur_idx).unwrap();
            let eq_name = fg.graph.node_weight(*eq_id).unwrap().name.as_ref().unwrap().clone();
            let eq_expr = Expr::Reference(Reference::Ref(eq_name));

            let (and_id, and_expr) = fg.add_primop2(
                PrimOp2Expr::And,
                *eq_id,
                eq_expr,
                fame5_inst_id,
                fame5_signal_ref,
            );

            // Connect AND result to ready signal of source node
            let node = fg.graph.node_weight(*cur_channel_id).unwrap();
            let ready_dst = Reference::RefDot(
                Box::new(Reference::Ref(node.name.as_ref().unwrap().clone())),
                signal_name.clone(),
            );
            fg.add_wire(and_id, and_expr, *cur_channel_id, Some(ready_dst));
        }
    }
}
