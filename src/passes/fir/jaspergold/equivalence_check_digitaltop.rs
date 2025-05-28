use indexmap::IndexSet;
use rusty_firrtl::{CircuitModule, DefName, Identifier};
use chirrtl_parser::parse_circuit;
use crate::passes::fir::jaspergold::equivalence_check::*;
use crate::passes::fir::modify_names::add_sfx_to_module_names;
use crate::passes::fir::to_ast::to_ast;
use crate::passes::ast::chirrtl_print::ChirrtlPrinter;
use crate::passes::ast::print::Printer;
use crate::common::RippleIRErr;
use crate::passes::runner::run_fir_passes_from_circuit;

pub fn equivalence_check_customtop(input_fir: &str, top: &str, clock: &str, reset: &str) -> Result<(), RippleIRErr> {
    let filename = format!("./test-inputs/{}.fir", input_fir);
    let source = std::fs::read_to_string(filename).expect("input file to exist");
    export_firrtl_and_sv("golden", input_fir, &source).expect("golden verilog export failed");

    let circuit = parse_circuit(&source).expect("firrtl parser");
    let mut ir = run_fir_passes_from_circuit(&circuit)?;

    let old_hier = ir.hier.clone();
    add_sfx_to_module_names(&mut ir, "_impl");

    let circuit_reconstruct = to_ast(&ir);
    let mut printer = ChirrtlPrinter::new();
    let circuit_str = printer.print_circuit(&circuit_reconstruct);
    export_firrtl_and_sv("impl", input_fir, &circuit_str).expect("impl verilog export failed");

    let mut top_sv_filename = verilog_outdir("golden", input_fir);
    top_sv_filename.push_str(&format!("/{top}.sv"));

    let all_childs: IndexSet<&Identifier> = old_hier.all_childs(
        &Identifier::Name(top.to_string()))
        .iter()
        .map(|id| old_hier.graph.node_weight(*id).unwrap().name())
        .collect();

    for cm in circuit.modules {
        if let CircuitModule::ExtModule(em) = cm.as_ref() {
            let DefName(name) = &em.defname;
            if !all_childs.contains(name) {
                continue;
            }
            if let Identifier::Name(x) = name {
                let input = format!("./test-inputs/{}.v", x);
                copy_file(&input,
                    &format!("{}/{}.sv", verilog_outdir("golden", input_fir), x))
                    .expect("golden verilog copy failed");
                copy_file(&input,
                    &format!("{}/{}.sv", verilog_outdir("impl", input_fir), x))
                    .expect("impl verilog copy failed");
            }
        }
    }

    export_tcl(&input_fir, top, clock, reset).expect("tcl export failed");

    let result = run_jaspergold(&input_fir, ".").expect("jaspergold run failed");
    match result {
        EquivStatus::Proven(x) => {
            println!("Proved {} properties", x);
            Ok(())
        }
        EquivStatus::NothingToProve => {
            println!("Nothing to prove...");
            Ok(())
        }
        EquivStatus::CounterExample(x) => {
            panic!("Found {} counter examples", x);
        }
        EquivStatus::Unknown(stdout) => {
            println!("{}", stdout);
            panic!("Found unknown");
        }
    }
}

#[cfg(test)]
mod test {
    use test_case::test_case;
    use crate::common::RippleIRErr;
    use super::*;

    #[test_case("chipyard.harness.TestHarness.RocketConfig", "GenericDeserializer_TLBeatw88_f32", "clock", "reset" ; "GenericDeser0")]
    #[test_case("chipyard.harness.TestHarness.RocketConfig", "GenericDeserializer_TLBeatw67_f32", "clock", "reset" ; "GenericDeser2")]
    #[test_case("chipyard.harness.TestHarness.RocketConfig", "GenericDeserializer_TLBeatw87_f32", "clock", "reset" ; "GenericDeser3")]
// #[test_case("chipyard.harness.TestHarness.RocketConfig", "SerialTL0ClockSinkDomain", "auto_clock_in_clock", "auto_clock_in_reset" ; "SerialTL0ClockSinkDomain")]
// #[test_case("chipyard.harness.TestHarness.RocketConfig", "Rocket", "clock", "reset" ; "Rocket")]
// #[test_case("chipyard.harness.TestHarness.RocketConfig", "TLSerdesser_serial_tl_0", "clock", "reset" ; "serial_tl_0")]
    fn run(name: &str, top: &str, clock: &str, reset: &str) -> Result<(), RippleIRErr> {
        equivalence_check_customtop(name, top, clock, reset)?;
        Ok(())
    }
}
