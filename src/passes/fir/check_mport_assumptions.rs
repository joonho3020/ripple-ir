use petgraph::{graph::NodeIndex, visit::EdgeRef, Direction::{Incoming, Outgoing}};
use indexmap::{IndexSet, IndexMap};
use rusty_firrtl::*;
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
        if mems_with_rwport.len() > 0 {
            ret.insert(name.clone(), mems_with_rwport);
        }
    }
    return ret;
}

fn mems_with_readwrite_port(fg: &FirGraph) -> Vec<Identifier> {
    let mut ret = vec![];
    for id in fg.graph.node_indices() {
        let node = fg.graph.node_weight(id).unwrap();
        match &node.nt {
            FirNodeType::SMem(..) => {
                if has_inferred_readwrite_port(fg, id) {
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

fn has_inferred_readwrite_port(fg: &FirGraph, id: NodeIndex) -> bool {
    let mem = fg.graph.node_weight(id).unwrap();
    let port_ids = fg.graph.neighbors_directed(id, Outgoing);

    let mut wport_en_drivers: Vec<Drivers> = vec![];
    let mut rport_en_drivers: Vec<Drivers> = vec![];

    let mut wport_addrs: Vec<Expr> = vec![];
    let mut rport_addrs: Vec<Expr> = vec![];

    fn add_drivers(fg: &FirGraph, port_id: NodeIndex) -> Option<Drivers> {
        match fg.parent_with_type(port_id, FirEdgeType::MemPortEn) {
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

    fn address(fg: &FirGraph, port_id: NodeIndex) -> Expr {
        let eid = fg.parent_with_type(port_id, FirEdgeType::MemPortAddr).unwrap();
        let edge = fg.graph.edge_weight(eid).unwrap();
        edge.src.clone()
    }

    for port_id in port_ids {
        let port = fg.graph.node_weight(port_id).unwrap();
        match &port.nt {
            FirNodeType::WriteMemPort(ppath) => {
                assert!(ppath.path.collect_sels().len() < 2);
                if let Some(drivers) = add_drivers(fg, port_id) {
                    wport_en_drivers.push(drivers);
                }
                wport_addrs.push(address(fg, port_id));
            }
            FirNodeType::ReadMemPort(ppath) => {
                assert!(ppath.path.collect_sels().len() < 2);
                if let Some(drivers) = add_drivers(fg, port_id) {
                    rport_en_drivers.push(drivers);
                }
                rport_addrs.push(address(fg, port_id));
            }
            FirNodeType::InferMemPort(ppath) => {
                assert!(ppath.path.collect_sels().len() < 2);
                todo!("Infer memory port type")
            }
            _ => {
                panic!("Memory {:?} connected to non port node {:?}", mem, port);
            }
        }
    }

    if (wport_addrs.len() != 1 || rport_addrs.len() != 1) ||
        (wport_en_drivers.len() != 1 || rport_en_drivers.len() != 1)
    {
        return false;
    }

    if wport_addrs[0] == rport_addrs[0] {
        return true;
    }

    let wport_drivers = &wport_en_drivers[0];
    let rport_drivers = &rport_en_drivers[0];

    let mut found_complement = false;
    'wport_loop: for w in wport_drivers {
        for r in rport_drivers {
            if check_complement(w, r) {
                found_complement = true;
                break 'wport_loop;
            }
        }
    }
    return found_complement;
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
            let teid = fg.parent_with_type(id, FirEdgeType::MuxTrue).unwrap();
            let feid = fg.parent_with_type(id, FirEdgeType::MuxFalse).unwrap();
            let ceid = fg.parent_with_type(id, FirEdgeType::MuxCond).unwrap();

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
        FirNodeType::Phi(..) => {
            // This is a bit more pessimistic than the FIRRTL version as
            // FIRRTL performs mux expansion and in theory can traverse
            // through more signals
            drivers.insert(cur_expr.clone());
        }
        FirNodeType::PrimOp2Expr(PrimOp2Expr::And) => {
            let op0_eid = fg.parent_with_type(id, FirEdgeType::Operand0).unwrap();
            let op0_edge = fg.graph.edge_weight(op0_eid).unwrap();
            let op0_ep = fg.graph.edge_endpoints(op0_eid).unwrap();

            let op1_eid = fg.parent_with_type(id, FirEdgeType::Operand1).unwrap();
            let op1_edge = fg.graph.edge_weight(op1_eid).unwrap();
            let op1_ep = fg.graph.edge_endpoints(op1_eid).unwrap();

            drivers.insert(cur_expr.clone());
            track_en_drivers(fg, &op0_edge.src, op0_ep.0, drivers);
            track_en_drivers(fg, &op1_edge.src, op1_ep.0, drivers);
        }
        FirNodeType::PrimOp2Expr(PrimOp2Expr::Eq) => {
            let op0_eid = fg.parent_with_type(id, FirEdgeType::Operand0).unwrap();
            let op0_edge = fg.graph.edge_weight(op0_eid).unwrap();

            let op1_eid = fg.parent_with_type(id, FirEdgeType::Operand1).unwrap();
            let op1_edge = fg.graph.edge_weight(op1_eid).unwrap();

            drivers.insert(cur_expr.clone());
            drivers.insert(Expr::PrimOp2Expr(
                    PrimOp2Expr::Eq,
                    Box::new(op0_edge.src.clone()),
                    Box::new(op1_edge.src.clone())));
        }
        FirNodeType::PrimOp1Expr(PrimOp1Expr::Not) => {
            let op0_eid = fg.parent_with_type(id, FirEdgeType::Operand0).unwrap();
            let op0_edge = fg.graph.edge_weight(op0_eid).unwrap();
            drivers.insert(cur_expr.clone());
            drivers.insert(Expr::PrimOp1Expr(
                    PrimOp1Expr::Not,
                    Box::new(op0_edge.src.clone())));
        }
        _ => {
            drivers.insert(cur_expr.clone());
            for pedge in fg.graph.edges_directed(id, Incoming) {
                let edge = fg.graph.edge_weight(pedge.id()).unwrap();

                if let Some(dst_ref) = &edge.dst {
                    // Connection, follow it
                    let dst_expr = Expr::Reference(dst_ref.clone());
                    if *cur_expr == dst_expr {
                        let ep = fg.graph.edge_endpoints(pedge.id()).unwrap();
                        track_en_drivers(fg, &edge.src, ep.0, drivers);
                    }
                } else if edge.et == FirEdgeType::PhiOut {
                    let phi_id = fg.graph.edge_endpoints(pedge.id()).unwrap().0;

                    // Phi with no selection signal. Added to set priority
                    // between partial connections between aggregate types.
                    // These are just normal connections so we should track
                    // connections that has the current expression as the destination.
                    if fg.parent_with_type(phi_id, FirEdgeType::PhiSel).is_none() {
                        let phi_parents = fg.graph.edges_directed(phi_id, Incoming);
                        for phi_parent in phi_parents {
                            let phi_iedge = fg.graph.edge_weight(phi_parent.id()).unwrap();
                            if let Some(dst_ref) = &phi_iedge.dst {
                                let dst_expr = Expr::Reference(dst_ref.clone());
                                if *cur_expr == dst_expr {
                                    let phi_ep = fg.graph.edge_endpoints(phi_parent.id()).unwrap();
                                    track_en_drivers(fg, &phi_iedge.src, phi_ep.0, drivers);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use indexmap::IndexMap;
    use rusty_firrtl::Identifier;
    use crate::{
        common::RippleIRErr,
        passes::fir::check_mport_assumptions::check_mport_assumptions,
        passes::runner::*,
    };
    use super::RWPortMap;

    /// Run the AST to graph conversion and export the graph form
    fn run(name: &str) -> Result<RWPortMap, RippleIRErr> {
        let fir = run_passes_from_filepath(&format!("./test-inputs/{}.fir", name))?;
        let mems_with_rwport = check_mport_assumptions(&fir);
        Ok(mems_with_rwport)
    }

    #[test]
    fn singleport_sram() -> Result<(), RippleIRErr> {
        let rwport_map = run("SinglePortSRAM")?;
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
        let rwport_map = run("chipyard.harness.TestHarness.RocketConfig")
            .expect("rocket failed");

        let mut expect = IndexMap::new();
        expect.insert(
            Identifier::Name("Directory".to_string()),
            vec![
                Identifier::Name("cc_dir".to_string())
            ]);

        expect.insert(
            Identifier::Name("BankedStore".to_string()),
            vec![
                Identifier::Name("cc_banks_0".to_string()),
                Identifier::Name("cc_banks_1".to_string()),
                Identifier::Name("cc_banks_2".to_string()),
                Identifier::Name("cc_banks_3".to_string()),
            ]);

        expect.insert(
            Identifier::Name("BankedStore".to_string()),
            vec![
                Identifier::Name("cc_banks_0".to_string()),
                Identifier::Name("cc_banks_1".to_string()),
                Identifier::Name("cc_banks_2".to_string()),
                Identifier::Name("cc_banks_3".to_string()),
            ]);

        expect.insert(
            Identifier::Name("DCacheDataArray".to_string()),
            vec![
                Identifier::Name("rockettile_dcache_data_arrays_0".to_string())
            ]);

        expect.insert(
            Identifier::Name("DCache".to_string()),
            vec![
                Identifier::Name("rockettile_dcache_tag_array".to_string())
            ]);

        expect.insert(
            Identifier::Name("ICache".to_string()),
            vec![
                Identifier::Name("rockettile_icache_tag_array".to_string()),
                Identifier::Name("rockettile_icache_data_arrays_0".to_string()),
                Identifier::Name("rockettile_icache_data_arrays_1".to_string())
            ]);

        expect.insert(
            Identifier::Name("TLRAM_ScratchpadBank".to_string()),
            vec![
                Identifier::Name("mem".to_string()),
            ]);

        assert_eq!(rwport_map, expect);
    }

    #[test]
    fn boom() {
        let rwport_map = run("chipyard.harness.TestHarness.LargeBoomV3Config")
            .expect("boom failed");

        let mut expect = IndexMap::new();
        expect.insert(
            Identifier::Name("Directory".to_string()),
            vec![
                Identifier::Name("cc_dir".to_string())
            ]);

        expect.insert(
            Identifier::Name("BankedStore".to_string()),
            vec![
                Identifier::Name("cc_banks_0".to_string()),
                Identifier::Name("cc_banks_1".to_string()),
                Identifier::Name("cc_banks_2".to_string()),
                Identifier::Name("cc_banks_3".to_string()),
                Identifier::Name("cc_banks_4".to_string()),
                Identifier::Name("cc_banks_5".to_string()),
                Identifier::Name("cc_banks_6".to_string()),
                Identifier::Name("cc_banks_7".to_string()),
            ]);

        expect.insert(
            Identifier::Name("ICache".to_string()),
            vec![
                Identifier::Name("tag_array".to_string()),
                Identifier::Name("dataArrayB0Way_0".to_string()),
                Identifier::Name("dataArrayB0Way_1".to_string()),
                Identifier::Name("dataArrayB0Way_2".to_string()),
                Identifier::Name("dataArrayB0Way_3".to_string()),
                Identifier::Name("dataArrayB0Way_4".to_string()),
                Identifier::Name("dataArrayB0Way_5".to_string()),
                Identifier::Name("dataArrayB0Way_6".to_string()),
                Identifier::Name("dataArrayB0Way_7".to_string()),
                Identifier::Name("dataArrayB1Way_0".to_string()),
                Identifier::Name("dataArrayB1Way_1".to_string()),
                Identifier::Name("dataArrayB1Way_2".to_string()),
                Identifier::Name("dataArrayB1Way_3".to_string()),
                Identifier::Name("dataArrayB1Way_4".to_string()),
                Identifier::Name("dataArrayB1Way_5".to_string()),
                Identifier::Name("dataArrayB1Way_6".to_string()),
                Identifier::Name("dataArrayB1Way_7".to_string()),
            ]);

        expect.insert(
            Identifier::Name("L1MetadataArray".to_string()),
            vec![
                Identifier::Name("tag_array".to_string()),
            ]);


        expect.insert(
            Identifier::Name("PTW".to_string()),
            vec![
                Identifier::Name("l2_tlb_ram".to_string()),
            ]);

        expect.insert(
            Identifier::Name("TLRAM_ScratchpadBank".to_string()),
            vec![
                Identifier::Name("mem".to_string()),
            ]);

        assert_eq!(rwport_map, expect);
    }
}
