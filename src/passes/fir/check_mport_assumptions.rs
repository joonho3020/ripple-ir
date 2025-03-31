use petgraph::{graph::{EdgeIndex, NodeIndex}, visit::EdgeRef, Direction::{Incoming, Outgoing}};
use indexmap::{IndexSet, IndexMap};
use chirrtl_parser::ast::*;
use crate::ir::fir::*;

pub type RWPortMap = IndexMap<Identifier, Vec<Identifier>>;

/// Check for Chisel SRAM port inference conditions:
/// - Merging read and write ports into a single readwrite port
///   - When read and write ports have mutually exclusive enable signals
///   - When read and write ports have the same address signal and the ruw behavior is undefined
///   and the read/write latencies are both equal to one
/// - Merging multiple read/write ports into a single port
///   - When a "mem.read" or "mem.write" is called under multiple when conditions and
///   the conditions are mutually exclusive
/// - We don't want port inference to happen. Make sure the above conditions don't exist in the
/// input CHIRRTL file
/// - In practice, read/write port merging doesn't happen. Only need to consider cases where
/// readwrite ports are generated
pub fn check_mport_assumptions(fir: &FirIR) -> RWPortMap {
    let mut ret = RWPortMap::new();
    for (name, fgraph) in fir.graphs.iter() {
        let mems_with_rwport = mems_with_readwrite_port(&fgraph);
        ret.insert(name.clone(), mems_with_rwport);
    }
    return ret;
}

fn mems_with_readwrite_port(fg: &FirGraph) -> Vec<Identifier> {
    let mut ret = vec![];
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
        match &node.nt {
            FirNodeType::SMem(..) => {
                if !has_readwrite_port(fg, id) {
                    ret.push(node.name.as_ref().unwrap().clone());
                }
            }
            FirNodeType::CMem => {
                // Cannot merge for CMem
                continue;
            }
            _ => {
                continue;
            }
        }
    }
    return ret;
}


type Drivers = IndexSet<Expr>;

fn has_readwrite_port(fg: &FirGraph, id: NodeIndex) -> bool {
    let mem = fg.graph.node_weight(id).unwrap();
    let port_ids = fg.graph.neighbors_directed(id, Outgoing);

    let mut wport_en_drivers: Vec<Drivers> = vec![];
    let mut rport_en_drivers: Vec<Drivers> = vec![];

    fn add_drivers(fg: &FirGraph, port_id: NodeIndex) -> Option<Drivers> {
        match parent_with_type(fg, port_id, FirEdgeType::MemPortEn) {
            Some(eid) => {
                let mut drivers = Drivers::new();
                let edge = fg.graph.edge_weight(eid).unwrap();
                let src_id = fg.graph.edge_endpoints(eid).unwrap().0;
                track_en_drivers(fg, &edge.src, src_id, &mut drivers);
                return Some(drivers)
            }
            None => {
                // No enable signal, cannot be merged
                return None;
            }
        }
    }

    for port_id in port_ids {
        let port = fg.graph.node_weight(port_id).unwrap();
        match &port.nt {
            FirNodeType::WriteMemPort(cond) => {
                assert!(cond.collect_sels().len() < 2);
                if let Some(drivers) = add_drivers(fg, port_id) {
                    wport_en_drivers.push(drivers);
                }
            }
            FirNodeType::ReadMemPort(cond) => {
                assert!(cond.collect_sels().len() < 2);
                if let Some(drivers) = add_drivers(fg, port_id) {
                    rport_en_drivers.push(drivers);
                }
            }
            FirNodeType::InferMemPort(cond) => {
                assert!(cond.collect_sels().len() < 2);
                todo!("Infer memory port type")
            }
            _ => {
                panic!("Memory {:?} connected to non port node {:?}", mem, port);
            }
        }
    }

    if wport_en_drivers.len() != 1 && rport_en_drivers.len() != 1 {
        return false;
    }

    let wport_drivers = &wport_en_drivers[0];
    let rport_drivers = &rport_en_drivers[0];

    for w in wport_drivers {
        let mut found_complement = false;
        for r in rport_drivers {
            if check_complement(w, r) {
                found_complement = true;
                break;
            }
        }

        if !found_complement {
            return false
        }
    }
    return true;
}

fn check_complement(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (_, Expr::PrimOp1Expr(PrimOp1Expr::Not, b_expr)) => {
            a == b_expr.as_ref()
        }
        (Expr::PrimOp1Expr(PrimOp1Expr::Not, a_expr), _) => {
            a_expr.as_ref() == b
        }
        (_, Expr::PrimOp2Expr(PrimOp2Expr::Eq, b_op0, b_op1)) => {
            (a == b_op0.as_ref() && expr_is_zero(b_op1.as_ref())) ||
            (expr_is_zero(b_op0.as_ref()) && a == b_op1.as_ref())
        }
        (Expr::PrimOp2Expr(PrimOp2Expr::Eq, a_op0, a_op1), _) => {
            (expr_is_zero(a_op0.as_ref()) && b == a_op1.as_ref()) ||
            (b == a_op0.as_ref() && expr_is_zero(a_op1.as_ref()))
        }
        _ => {
            false
        }
    }
}

fn parent_with_type(fg: &FirGraph, id: NodeIndex, et: FirEdgeType) -> Option<EdgeIndex> {
    for eid in fg.graph.edges_directed(id, Incoming) {
        let edge = fg.graph.edge_weight(eid.id()).unwrap();
        if edge.et == et {
            return Some(eid.id());
        }
    }
    return None;
}

fn expr_is_literal(expr: &Expr) -> Option<Int> {
    match expr {
        Expr::SIntInit(_, x) |
            Expr::UIntInit(_, x) => Some(x.clone()),
        _ => None
    }
}

fn expr_is_zero(expr: &Expr) -> bool {
    if let Some(x) = expr_is_literal(expr) {
        x == Int::from(0)
    } else {
        false
    }
}

fn is_literal(node: &FirNode, value: Int) -> bool {
    match &node.nt {
        FirNodeType::UIntLiteral(_, x) |
            FirNodeType::SIntLiteral(_, x) => {
            *x == value
        }
        _ => { false }
    }
}

fn is_one(node: &FirNode) -> bool {
    is_literal(node, Int::from(1))
}

fn is_zero(node: &FirNode) -> bool {
    is_literal(node, Int::from(0))
}

fn track_en_drivers(fg: &FirGraph, cur_expr: &Expr, id: NodeIndex, drivers: &mut Drivers) {
    let node = fg.graph.node_weight(id).unwrap();
    match node.nt {
        FirNodeType::Mux => {
            let teid = parent_with_type(fg, id, FirEdgeType::MuxTrue).unwrap();
            let feid = parent_with_type(fg, id, FirEdgeType::MuxFalse).unwrap();
            let ceid = parent_with_type(fg, id, FirEdgeType::MuxCond).unwrap();

            let tep = fg.graph.edge_endpoints(teid).unwrap();
            let tedge = fg.graph.edge_weight(teid).unwrap();
            let tval = fg.graph.node_weight(tep.0).unwrap();

            let fep = fg.graph.edge_endpoints(feid).unwrap();
            let fval = fg.graph.node_weight(fep.0).unwrap();

            let cedge = fg.graph.edge_weight(ceid).unwrap();
            let cep = fg.graph.edge_endpoints(ceid).unwrap();
            if is_one(tval) && is_zero(fval) {
                track_en_drivers(fg, &cedge.src, cep.0, drivers);
            } else if is_zero(fval) {
                track_en_drivers(fg, &cedge.src, cep.0, drivers);
                track_en_drivers(fg, &tedge.src, tep.0, drivers);
            } else {
                drivers.insert(cur_expr.clone());
            }
        }
        FirNodeType::Phi => {
            // NOTE: This is a bit more pessimistic than the FIRRTL version
            drivers.insert(cur_expr.clone());
        }
        FirNodeType::PrimOp2Expr(PrimOp2Expr::And) => {
            let op0_eid = parent_with_type(fg, id, FirEdgeType::Operand0).unwrap();
            let op0_edge = fg.graph.edge_weight(op0_eid).unwrap();
            let op0_ep = fg.graph.edge_endpoints(op0_eid).unwrap();

            let op1_eid = parent_with_type(fg, id, FirEdgeType::Operand1).unwrap();
            let op1_edge = fg.graph.edge_weight(op1_eid).unwrap();
            let op1_ep = fg.graph.edge_endpoints(op1_eid).unwrap();

            drivers.insert(cur_expr.clone());
            track_en_drivers(fg, &op0_edge.src, op0_ep.0, drivers);
            track_en_drivers(fg, &op1_edge.src, op1_ep.0, drivers);
        }
        _ => {
            drivers.insert(cur_expr.clone());
        }
    }
}

#[cfg(test)]
mod test {
    use indexmap::IndexMap;
    use chirrtl_parser::ast::Identifier;

    use crate::{
        common::graphviz::GraphViz,
        common::RippleIRErr,
        passes::fir::check_mport_assumptions::check_mport_assumptions,
        passes::runner::*,
    };

    use super::RWPortMap;

    /// Run the AST to graph conversion and export the graph form
    fn run(name: &str, export: bool) -> Result<RWPortMap, RippleIRErr> {
        let fir = run_passes_from_filepath(&format!("./test-inputs/{}.fir", name))?;
        let mems_with_rwport = check_mport_assumptions(&fir);
        if export {
            for (sub_name, graph) in fir.graphs {
                graph.export_graphviz(
                    &format!("./test-outputs/{}-{}.remove_phi.dot.pdf", name, sub_name),
                    None, None, false)?;
            }
        }
        Ok(mems_with_rwport)
    }

    #[test]
    fn singleport_sram() -> Result<(), RippleIRErr> {
        let rwport_map = run("SinglePortSRAM", true)?;
        let mut expect = IndexMap::new();
        expect.insert(
            Identifier::Name("SinglePortSRAM".to_string()),
            vec![
                Identifier::Name("mem".to_string())
            ]);
        assert_eq!(rwport_map, expect);
        Ok(())
    }

    #[test]
    fn rocket() {
        run("chipyard.harness.TestHarness.RocketConfig", false)
            .expect("rocket failed");
    }

    #[test]
    fn boom() {
        run("chipyard.harness.TestHarness.LargeBoomV3Config", false)
            .expect("boom failed");
    }
}
