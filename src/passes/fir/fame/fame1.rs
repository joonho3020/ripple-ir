use crate::ir::fir::{FirEdge, FirEdgeType, FirGraph, FirIR, FirNode, FirNodeType};
use crate::ir::typetree::tnode::{GroundType, TypeDirection, TypeTreeNodeType};
use crate::ir::typetree::typetree::TypeTree;
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






// get inputs
// get outputs
// create new top
// create FSM
// change clock of the patient SSM to enable


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

        Ok(())
    }
}
