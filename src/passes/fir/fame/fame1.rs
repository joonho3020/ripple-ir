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

fn create_and_reductiont_tree(dcpld_inputs: &Vec<NodeIndex>, fg: &mut FirGraph) -> Option<NodeIndex> {
    if dcpld_inputs.is_empty() {
        return None;
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
            None
        );
        let valid_id = fg.graph.add_node(valid_node);

        let edge = FirEdge::new(valid_ref, Some(Reference::Ref(wire_name)), FirEdgeType::Wire);
        fg.graph.add_edge(input_id, valid_id, edge);

        valid_id
    }).collect();

    // Build reduction tree
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

    let mut clock: Option<TypeTree> = None;
    let mut ichans: IndexMap<Identifier, TypeTree> = IndexMap::new();
    let mut ochans: IndexMap<Identifier, TypeTree> = IndexMap::new();

    let leaves = io_ttree.leaves();
    for leaf in leaves {
        let x = io_ttree.get_node(leaf).unwrap();
        let port_ttree = io_ttree.subtree_from_id(leaf).clone_ttree();

        if x.dir == TypeDirection::Incoming {
            ochans.insert(x.name.unwrap(), port_ttree);
        } else {
            match x.tpe {
                TypeTreeNodeType::Ground(GroundType::Clock) => {
                    clock = Some(port_ttree);
                }
                _ => {
                    ichans.insert(x.name.unwrap(), port_ttree);
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

    let clock_node = FirNode::new(
        Some(Identifier::Name("host_clock".to_string())),
        FirNodeType::Input,
        clock);

    let host_clock_id = fame_top.graph.add_node(clock_node);

    let mut ichan_map: IndexMap<Identifier, NodeIndex> = IndexMap::new();

    for (name, ttree) in ichans {
        let dcpld_ttree = ttree.decoupled(TypeDirection::Outgoing);
        println!("name {:?}", name);
        dcpld_ttree.view().unwrap().print_tree();
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

    let all_input_valid = create_and_reductiont_tree(
        &ichan_map.iter().map(|(_, v)| *v).collect(),
        &mut fame_top).unwrap();

    let rir = from_fir(&fir);
    let comb_deps = combinational_analaysis(&rir);


// get inputs
// get outputs
// create FSM
// change clock of the patient SSM to enable

    fir.add_module(Identifier::Name("FameTop".to_string()), fame_top);

}


#[cfg(test)]
mod test {
    use super::*;
    use crate::common::RippleIRErr;
    use crate::passes::runner::run_fir_passes;
    use chirrtl_parser::parse_circuit;

    #[test]
    fn run() -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string("./test-inputs/GCD.fir")?;
        let circuit = parse_circuit(&source).expect("firrtl parser");
        let mut fir = run_fir_passes(&circuit)?;

        fame1_transform(&mut fir);

        fir.export("./test-outputs", "fame")?;

        Ok(())
    }
}
