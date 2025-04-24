use crate::ir::fir::{FirEdge, FirEdgeType, FirGraph, FirIR, FirNode, FirNodeType};
use crate::ir::typetree::tnode::{GroundType, TypeDirection, TypeTreeNodeType};
use crate::ir::typetree::typetree::TypeTree;
use crate::passes::rir::combinational::combinational_analaysis;
use crate::passes::rir::from_fir::from_fir;
use chirrtl_parser::ast::*;
use indexmap::IndexMap;
use petgraph::graph::NodeIndex;

impl TypeTree {
    fn decoupled(&self, dir: TypeDirection) -> Self {
        let tpe = self.to_type();
        let uint1 = Type::TypeGround(TypeGround::UInt(Some(Width(1))));

        let bits = Field::Straight(Identifier::Name("bits".to_string()), Box::new(tpe));
        let valid = Field::Straight(Identifier::Name("valid".to_string()), Box::new(uint1.clone()));
        let ready = Field::Flipped(Identifier::Name("ready".to_string()), Box::new(uint1.clone()));

        let fields = TypeAggregate::Fields(Box::new(vec![Box::new(bits), Box::new(valid), Box::new(ready)]));
        let dcpld = Type::TypeAggregate(Box::new(fields));

        Self::build_from_type(&dcpld, dir)
    }
}
fn build_tree(signals: &[NodeIndex], fg: &mut FirGraph) -> NodeIndex {
    if signals.len() == 1 {
        return signals[0];
    }

    let mid = signals.len() / 2;
    let left = build_tree(&signals[..mid], fg);
    let right = build_tree(&signals[mid..], fg);

    let and_node = FirNode::new(
        Some(fg.namespace.next()),
        FirNodeType::PrimOp2Expr(PrimOp2Expr::And),
        None
    );
    let and_id = fg.graph.add_node(and_node);

    let lname = fg.graph.node_weight(left).unwrap().name.as_ref().unwrap().clone();
    let rname = fg.graph.node_weight(right).unwrap().name.as_ref().unwrap().clone();
    let left_ref = Expr::Reference(Reference::Ref(lname));
    let right_ref = Expr::Reference(Reference::Ref(rname));

    fg.graph.add_edge(left, and_id, FirEdge::new(left_ref, None, FirEdgeType::Operand0));
    fg.graph.add_edge(right, and_id, FirEdge::new(right_ref, None, FirEdgeType::Operand1));

    and_id
}

fn dcpld_valid_reduction_tree(dcpld_inputs: &Vec<NodeIndex>, fg: &mut FirGraph) -> Option<NodeIndex> {
    if dcpld_inputs.is_empty() {
        let one = FirNode::new(None,
            FirNodeType::UIntLiteral(Width(1), Int::from(1)),
            Some(TypeTree::build_from_ground_type(GroundType::UInt(Some(Width(1))))));
        let one_id = fg.graph.add_node(one);
        return Some(one_id);
    }

    // Extract valid signals from decoupled inputs
    let valid_signals: Vec<NodeIndex> = dcpld_inputs.iter().map(|&input_id| {
        let node = fg.graph.node_weight(input_id).unwrap();
        let valid_ref = Expr::Reference(Reference::RefDot(
            Box::new(Reference::Ref(node.name.as_ref().unwrap().clone())),
            Identifier::Name("valid".to_string())
        ));

        let wire_name = fg.namespace.next();
        let valid_node = FirNode::new(
            Some(wire_name.clone()),
            FirNodeType::Wire,
            Some(TypeTree::build_from_ground_type(GroundType::UInt(Some(Width(1)))))
        );
        let valid_id = fg.graph.add_node(valid_node);

        let edge = FirEdge::new(valid_ref, Some(Reference::Ref(wire_name)), FirEdgeType::Wire);
        fg.graph.add_edge(input_id, valid_id, edge);

        valid_id
    }).collect();

    // Build reduction tree
    if valid_signals.len() == 1 {
        Some(valid_signals[0])
    } else {
        Some(build_tree(&valid_signals, fg))
    }
}

fn reduction_tree(inputs: &Vec<NodeIndex>, fg: &mut FirGraph) -> Option<NodeIndex> {
    if inputs.is_empty() {
        return None;
    }

    // Extract signals from reference
    let valid_signals: Vec<NodeIndex> = inputs.iter().map(|&input_id| {
        let node = fg.graph.node_weight(input_id).unwrap();
        let signal_ref = Expr::Reference(Reference::Ref(node.name.as_ref().unwrap().clone()));

        let wire_name = fg.namespace.next();
        let valid_node = FirNode::new(
            Some(wire_name.clone()),
            FirNodeType::Wire,
            Some(TypeTree::build_from_ground_type(GroundType::UInt(Some(Width(1)))))
        );
        let valid_id = fg.graph.add_node(valid_node);

        let edge = FirEdge::new(signal_ref, Some(Reference::Ref(wire_name)), FirEdgeType::Wire);
        fg.graph.add_edge(input_id, valid_id, edge);

        valid_id
    }).collect();

    // Build reduction tree
    if valid_signals.len() == 1 {
        Some(valid_signals[0])
    } else {
        Some(build_tree(&valid_signals, fg))
    }
}

pub fn fame1_transform(fir: &mut FirIR) {
    let top_name = fir.hier.graph.node_weight(fir.hier.top().unwrap()).unwrap();
    let top = fir.graphs.get(top_name.name()).unwrap();

    let io_ttree_ = top.io_typetree();
    let io_ttree = io_ttree_.view().unwrap();

    let mut clock: Option<(Identifier, TypeTree)> = None;
    let mut ichans: IndexMap<Identifier, TypeTree> = IndexMap::new();
    let mut ochans: IndexMap<Identifier, TypeTree> = IndexMap::new();

    let leaves_with_path = io_ttree.leaves_with_path();

    for (path, leaf) in leaves_with_path {
        let x = io_ttree.get_node(leaf).unwrap();
        let port_ttree = io_ttree.subtree_from_id(leaf).clone_ttree();

        if x.dir == TypeDirection::Incoming {
            ochans.insert(path.name().unwrap(), port_ttree);
        } else {
            match x.tpe {
                TypeTreeNodeType::Ground(GroundType::Clock) => {
                    clock = Some((path.name().unwrap(), port_ttree));
                }
                _ => {
                    ichans.insert(path.name().unwrap(), port_ttree);
                }
            }
        }
    }

    assert!(clock.is_some(), "PatientSSM doesn't have a clock input");

    let mut fame_top = FirGraph::new(false);

    let patient_ssm_id = fame_top.graph.add_node(FirNode::new(
            Some(top_name.name().clone()),
            FirNodeType::Inst(top_name.name().clone()),
            None));

    let host_clock_name = Identifier::Name("host_clock".to_string());
    let clock_node = FirNode::new(
        Some(host_clock_name.clone()),
        FirNodeType::Input,
        Some(clock.as_ref().unwrap().1.clone()));

    let host_clock_id = fame_top.graph.add_node(clock_node);

    let mut ichan_map: IndexMap<Identifier, NodeIndex> = IndexMap::new();

    for (name, ttree) in ichans {
        let dcpld_ttree = ttree.decoupled(TypeDirection::Outgoing);
        let ichan_port = FirNode::new(
            Some(name.clone()),
            FirNodeType::Input,
            Some(dcpld_ttree));

        let id = fame_top.graph.add_node(ichan_port);
        ichan_map.insert(name.clone(), id);

        let src = Expr::Reference(
            Reference::RefDot(
                Box::new(Reference::Ref(name.clone())),
                Identifier::Name("bits".to_string())));

        let dst = Reference::RefDot(
            Box::new(Reference::Ref(top_name.name().clone())),
            name.clone());

        let idata_conn = FirEdge::new(src, Some(dst), FirEdgeType::Wire);
        fame_top.graph.add_edge(id, patient_ssm_id, idata_conn);
    }

    let rir = from_fir(&fir);
    let hier_comb_deps = combinational_analaysis(&rir);

    let comb_deps = hier_comb_deps.get(top_name.name()).unwrap();

    // Get module outputs and create decoupled interfaces with control logic
    let mut ochan_map: IndexMap<Identifier, NodeIndex> = IndexMap::new();
    let mut fired_or_firing_nodes = Vec::new();
    let mut fired_mux_nodes = Vec::new();

    for (name, ttree) in &ochans {
        // Create output port
        let dcpld_ttree = ttree.decoupled(TypeDirection::Incoming);
        let ochan_port = FirNode::new(
            Some(name.clone()),
            FirNodeType::Output,
            Some(dcpld_ttree)
        );
        let ochan_id = fame_top.graph.add_node(ochan_port);
        ochan_map.insert(name.clone(), ochan_id);

        // Connect data from patient SSM to output port
        let src = Reference::RefDot(
            Box::new(Reference::Ref(top_name.name().clone())),
            name.clone());
        let dst = Reference::RefDot(
            Box::new(Reference::Ref(name.clone())),
            Identifier::Name("bits".to_string()));

        let odata_conn = FirEdge::new(
            Expr::Reference(src),
            Some(dst),
            FirEdgeType::Wire
        );
        fame_top.graph.add_edge(patient_ssm_id, ochan_id, odata_conn);

        // Reduction tree for per output combinational connected logic
        let comb_dep_iports = comb_deps.get(name).unwrap();
        let comb_dep_ichan_map = ichan_map.iter()
            .filter(|(k, _)| comb_dep_iports.contains(*k))
            .map(|(_, v)| *v).collect();

        let ccls_valid_id = dcpld_valid_reduction_tree(&comb_dep_ichan_map, &mut fame_top).unwrap();

        // Create per-output fired and firing registers
        let fired_name = Identifier::Name(format!("{}_fired", name.to_string()));
        let fired_node = FirNode::new(
            Some(fired_name.clone()),
            FirNodeType::Reg,
            Some(TypeTree::build_from_ground_type(GroundType::UInt(Some(Width(1)))))
        );
        let fired_id = fame_top.graph.add_node(fired_node);

        let clk_edge = FirEdge::new(
            Expr::Reference(Reference::Ref(host_clock_name.clone())),
            None,
            FirEdgeType::Clock);

        fame_top.graph.add_edge(host_clock_id, fired_id, clk_edge);

        // Create not of the fired register output
        let not_fired_name = Identifier::Name(format!("{}_not_fired", name.to_string()));
        let not_fired_node = FirNode::new(
            Some(not_fired_name.clone()),
            FirNodeType::PrimOp1Expr(PrimOp1Expr::Not),
            None
        );
        let not_fired_id = fame_top.graph.add_node(not_fired_node);

        // Connect fired to not-fired
        let edge = FirEdge::new(
            Expr::Reference(Reference::Ref(fired_name.clone())),
            None,
            FirEdgeType::Operand0);

        fame_top.graph.add_edge(fired_id, not_fired_id, edge);

        // And the not-fired with ccls_input_valid
        let out_valid_name = Identifier::Name(format!("{}_valid", name.to_string()));
        let out_valid = FirNode::new(
            Some(out_valid_name.clone()),
            FirNodeType::PrimOp2Expr(PrimOp2Expr::And),
            None);

        let out_valid_id = fame_top.graph.add_node(out_valid);

        let ccls_input_valid_name_opt = fame_top.graph.node_weight(ccls_valid_id).unwrap().name.as_ref();
        let src_expr = match ccls_input_valid_name_opt {
            // Output channel has at least one combinationally coupled input channel
            Some(name) => Expr::Reference(Reference::Ref(name.clone())),
            // Output has no combinationally coupled input channel
            None => Expr::Reference(Reference::Ref(name.clone())),
        };

        let op0 = FirEdge::new(
            src_expr,
            None,
            FirEdgeType::Operand0);

        fame_top.graph.add_edge(ccls_valid_id, out_valid_id, op0);

        let op1 = FirEdge::new(
            Expr::Reference(Reference::Ref(not_fired_name.clone())),
            None,
            FirEdgeType::Operand1);

        fame_top.graph.add_edge(not_fired_id, out_valid_id, op1);

        // Connect out_valid to decoupled output port valid
        let edge = FirEdge::new(
            Expr::Reference(Reference::Ref(out_valid_name.clone())),
            Some(Reference::RefDot(
                    Box::new(Reference::Ref(name.clone())),
                    Identifier::Name("valid".to_string()))),
            FirEdgeType::Wire);
        fame_top.graph.add_edge(out_valid_id, ochan_id, edge);

        // Create Firing logic
        let firing_name = Identifier::Name(format!("{}_firing", name.to_string()));
        let firing = FirNode::new(
            Some(firing_name.clone()),
            FirNodeType::PrimOp2Expr(PrimOp2Expr::And),
            None);
        let firing_id = fame_top.graph.add_node(firing);

        let op0 = FirEdge::new(
            Expr::Reference(Reference::Ref(out_valid_name)),
            None,
            FirEdgeType::Operand0);

        fame_top.graph.add_edge(out_valid_id, firing_id, op0);

        let op1 = FirEdge::new(
            Expr::Reference(Reference::RefDot(
                    Box::new(Reference::Ref(name.clone())),
                    Identifier::Name("ready".to_string()))),
            None,
            FirEdgeType::Operand1);

        fame_top.graph.add_edge(ochan_id, firing_id, op1);

        // Create per-output firedOrFiring logic
        let fired_or_firing_name = Identifier::Name(format!("{}_firedOrFiring", name.to_string()));
        let fired_or_firing = FirNode::new(
            Some(fired_or_firing_name.clone()),
            FirNodeType::PrimOp2Expr(PrimOp2Expr::Or),
            None
        );
        let fired_or_firing_id = fame_top.graph.add_node(fired_or_firing);
        fired_or_firing_nodes.push(fired_or_firing_id);

        let op0 = FirEdge::new(
            Expr::Reference(Reference::Ref(fired_name.clone())),
            None,
            FirEdgeType::Operand0);

        fame_top.graph.add_edge(fired_id, fired_or_firing_id, op0);

        let op1 = FirEdge::new(
            Expr::Reference(Reference::Ref(firing_name.clone())),
            None,
            FirEdgeType::Operand1);

        fame_top.graph.add_edge(firing_id, fired_or_firing_id, op1);


        // Create mux that drives the Fired register
        let mux_name = Identifier::Name(format!("{}_fired_mux", name.to_string()));
        let mux = FirNode::new(Some(mux_name.clone()), FirNodeType::Mux, None);
        let mux_id = fame_top.graph.add_node(mux);
        fired_mux_nodes.push(mux_id);

        let op0 = FirEdge::new(
            Expr::Reference(Reference::Ref(fired_or_firing_name.clone())),
            None,
            FirEdgeType::MuxFalse);

        fame_top.graph.add_edge(fired_or_firing_id, mux_id, op0);

        let zero = FirNode::new(None, FirNodeType::UIntLiteral(Width(1), Int::from(0)), None);
        let zero_id = fame_top.graph.add_node(zero);
        let op1 = FirEdge::new(
            Expr::UIntInit(Width(1), Int::from(0)),
            None,
            FirEdgeType::MuxTrue);
        fame_top.graph.add_edge(zero_id, mux_id, op1);

        // Connect mux to fired register
        let edge = FirEdge::new(
            Expr::Reference(Reference::Ref(mux_name.clone())),
            Some(Reference::Ref(fired_name.clone())),
            FirEdgeType::Wire);
        fame_top.graph.add_edge(mux_id, fired_id, edge);
    }

    // And reduce all decoupled input port valid signals
    let all_input_valid_id = dcpld_valid_reduction_tree(
        &ichan_map.iter().map(|(_, v)| *v).collect(),
        &mut fame_top).unwrap();

    // And reduce all fired or firing signasl
    let all_fired_or_firing_id = reduction_tree(
        &fired_or_firing_nodes,
        &mut fame_top).unwrap();

    // Cycle finishing signal
    let cycle_finishing_name = Identifier::Name("cycle_finishing".to_string());
    let cycle_finishing = FirNode::new(
        Some(cycle_finishing_name.clone()),
        FirNodeType::PrimOp2Expr(PrimOp2Expr::And),
        None);

    let cycle_finishing_id = fame_top.graph.add_node(cycle_finishing);

    let all_input_valid = fame_top.graph.node_weight(all_input_valid_id).unwrap();
    let all_input_valid_name = all_input_valid.name.as_ref().unwrap();
    let op0 = FirEdge::new(
        Expr::Reference(Reference::Ref(all_input_valid_name.clone())),
        None,
        FirEdgeType::Operand0);

    fame_top.graph.add_edge(all_input_valid_id, cycle_finishing_id, op0);

    let all_fired_or_firing = fame_top.graph.node_weight(all_fired_or_firing_id).unwrap();
    let all_fired_or_firing_name = all_fired_or_firing.name.as_ref().unwrap();
    let op1 = FirEdge::new(
        Expr::Reference(Reference::Ref(all_fired_or_firing_name.clone())),
        None,
        FirEdgeType::Operand1);

    fame_top.graph.add_edge(all_fired_or_firing_id, cycle_finishing_id, op1);

    // Connect cycle_finishing to mux selectors
    for mux_id in fired_mux_nodes.iter() {
        let sel = FirEdge::new(
            Expr::Reference(Reference::Ref(cycle_finishing_name.clone())),
            None,
            FirEdgeType::MuxCond);
        fame_top.graph.add_edge(cycle_finishing_id, *mux_id, sel);
    }

    // Connect cycle_finishing to ready signals of input channels
    for (name, ichan_id) in ichan_map.iter() {
        let rdy = FirEdge::new(
            Expr::Reference(Reference::Ref(cycle_finishing_name.clone())),
            Some(Reference::RefDot(
                    Box::new(Reference::Ref(name.clone())),
                    Identifier::Name("ready".to_string()))),
            FirEdgeType::Wire);
        fame_top.graph.add_edge(cycle_finishing_id, *ichan_id, rdy);
    }

    // Connect cycle_finishing to clock signal of top
    let ssm_clk = FirEdge::new(
            Expr::Reference(Reference::Ref(cycle_finishing_name.clone())),
            Some(Reference::RefDot(
                    Box::new(Reference::Ref(top_name.name().clone())),
                    clock.unwrap().0)),
            FirEdgeType::Wire);
    fame_top.graph.add_edge(cycle_finishing_id, patient_ssm_id, ssm_clk);

    // Add the module to FIR
    fir.add_module(Identifier::Name("FameTop".to_string()), fame_top);
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::common::RippleIRErr;
    use crate::common::export_circuit;
    use crate::passes::fir::to_ast::to_ast;
    use crate::passes::runner::{run_fir_passes, run_fir_passes_from_circuit};
    use crate::passes::ast::print::Printer;
    use chirrtl_parser::parse_circuit;

    #[test]
    fn run() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/GCD.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");
        let mut fir = run_fir_passes_from_circuit(&circuit)?;

        fame1_transform(&mut fir);
        run_fir_passes(&mut fir)?;
        fir.export("./test-outputs", "fame")?;

        let transformed_circuit = to_ast(&fir);

        let firrtl = format!("./test-outputs/GCD.fir");

        let mut printer = Printer::new();
        let circuit_str = printer.print_circuit(&transformed_circuit);
        std::fs::write(&firrtl, circuit_str)?;
// export_circuit(&firrtl, &format!("test-outputs/{}/verilog", "GCD"))?;

        Ok(())
    }
}
