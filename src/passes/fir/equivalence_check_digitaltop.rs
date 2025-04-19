use indexmap::IndexSet;
use std::fs::File;
use std::io::Write;
use chirrtl_parser::ast::{CircuitModule, DefName, Identifier};
use chirrtl_parser::parse_circuit;
use crate::passes::fir::equivalence_check::*;
use crate::passes::fir::from_ast::from_circuit;
use crate::passes::fir::remove_unnecessary_phi::remove_unnecessary_phi;
use crate::passes::fir::check_phi_nodes::check_phi_node_connections;
use crate::passes::fir::modify_names::add_sfx_to_module_names;
use crate::passes::fir::to_ast::to_ast;
use crate::passes::ast::print::Printer;
use crate::common::RippleIRErr;

pub fn equivalence_check_digitaltop(input_fir: &str) -> Result<(), RippleIRErr> {
    let filename = format!("./test-inputs/{}.fir", input_fir);
    let source = std::fs::read_to_string(filename)?;
    export_firrtl_and_sv("golden", input_fir, &source)?;

    let circuit = parse_circuit(&source).expect("firrtl parser");
    let mut ir = from_circuit(&circuit);
    remove_unnecessary_phi(&mut ir);
    check_phi_node_connections(&ir)?;

    let old_hier = ir.hier.clone();
    add_sfx_to_module_names(&mut ir, "_impl");

    let circuit_reconstruct = to_ast(&ir);
    let mut printer = Printer::new();
    let circuit_str = printer.print_circuit(&circuit_reconstruct);
    export_firrtl_and_sv("impl", input_fir, &circuit_str)?;

    let mut top_sv_filename = verilog_outdir("golden", input_fir);
    top_sv_filename.push_str("/DigitalTop.sv");
    export_miter_digitaltop(&input_fir, &top_sv_filename)?;

    let all_childs: IndexSet<&Identifier> = old_hier.all_childs(
        &Identifier::Name("DigitalTop".to_string()))
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
                    &format!("{}/{}.sv", verilog_outdir("golden", input_fir), x))?;
                copy_file(&input,
                    &format!("{}/{}.sv", verilog_outdir("impl", input_fir), x))?;
            }
        }
    }

    export_tcl(&input_fir, "DigitalTop")?;

    let result = run_jaspergold(&input_fir, ".")?;
    match result {
        EquivStatus::Proven => {
            Ok(())
        }
        EquivStatus::NothingToProve => {
            println!("Nothing to prove...");
            Ok(())
        }
        EquivStatus::CounterExample => {
            panic!("Found counter example");
        }
        EquivStatus::Unknown => {
            panic!("Found unknown");
        }
    }
}

fn export_miter_digitaltop(firname: &str, top_sv_filename: &str) -> Result<(), RippleIRErr> {
    let top_module = std::fs::read_to_string(top_sv_filename)?;
    let module = parse_module(top_module.lines()).expect("Failed to parse module");

    let miter_content = generate_miter_digitaltop(&module);
    let mut file = File::create(&miter_filename(firname))?;
    file.write_all(miter_content.as_bytes())?;
    Ok(())
}

fn generate_miter_digitaltop(module: &Module) -> String {
    let mut ret = String::new();

    // Module declaration
    ret.push_str("module top_miter(\n");
    ret.push_str("  input clock, reset");

    let top_clock = "auto_chipyard_prcictrl_domain_reset_setter_clock_in_member_allClocks_uncore_clock";
    let top_reset = "auto_chipyard_prcictrl_domain_reset_setter_clock_in_member_allClocks_uncore_reset";

    // Add other inputs
    for input in &module.inputs {
        if input.name != top_clock && input.name != top_reset  {
            ret.push_str(",\n  input ");
            if let Some(ref width) = input.width {
                ret.push_str(&format!("{}", width));
                ret.push(' ');
            }
            ret.push_str(&input.name);
        }
    }

    ret.push_str(",\n  output equiv\n);\n");

    // Wire declarations for each output
    for (_, oport) in module.outputs.iter().enumerate() {
        ret.push_str("  wire ");
        if let Some(ref width) = oport.width {
            ret.push_str(&format!("{}", width));
        }
        ret.push_str(&format!(" {}_1, {}_2;\n", oport.name, oport.name));
    }
    ret.push_str("\n");

    // Instantiate reference design
    ret.push_str("  // Instantiate reference design\n");
    ret.push_str(&format!("  {} ref_inst (\n", module.name));
    ret.push_str(&format!("    .{}(clock),\n", top_clock));
    ret.push_str(&format!("    .{}(reset),\n", top_reset));
    let has_oport = !module.outputs.is_empty();
    let iport_cnt = module.inputs.iter().count();
    for (i, input) in module.inputs.iter().enumerate() {
        if input.name != top_clock && input.name != top_reset {
            ret.push_str(&format!("    .{}({})", input.name, input.name));
            if has_oport || (iport_cnt - 1 != i) {
                ret.push_str(",\n");
            }
        }
    }
    for (i, oport) in module.outputs.iter().enumerate() {
        let comma = if i == module.outputs.len() - 1 { "" } else { "," };
        ret.push_str(&format!("    .{}({}_1){}\n", oport.name, oport.name, comma));
    }
    ret.push_str("  );\n\n");

    // Instantiate implementation design
    ret.push_str("  // Instantiate implementation design\n");
    ret.push_str(&format!("  {}_impl impl_inst (\n", module.name));
    ret.push_str(&format!("    .{}(clock),\n", top_clock));
    ret.push_str(&format!("    .{}(reset),\n", top_reset));
    for (i, input) in module.inputs.iter().enumerate() {
        if input.name != top_clock && input.name != top_reset {
            ret.push_str(&format!("    .{}({})", input.name, input.name));
            if has_oport || (iport_cnt - 1 != i) {
                ret.push_str(",\n");
            }
        }
    }
    for (i, oport) in module.outputs.iter().enumerate() {
        let comma = if i == module.outputs.len() - 1 { "" } else { "," };
        ret.push_str(&format!("    .{}({}_2){}\n", oport.name, oport.name, comma));
    }
    ret.push_str("  );\n\n");

    // Property and assertion for each output
    for oport in &module.outputs {
        ret.push_str(&format!("  property {}_match;\n", oport.name));
        ret.push_str(&format!(
            "    @(posedge clock) disable iff (!reset) ({}_1 == {}_2);\n",
            oport.name, oport.name
        ));
        ret.push_str("  endproperty\n");
        ret.push_str(&format!("  assert property ({}_match);\n\n", oport.name));
    }

    ret.push_str("endmodule\n");

    ret
}

#[cfg(test)]
mod test {
    use test_case::test_case;
    use crate::common::RippleIRErr;
    use super::*;

    #[test_case("chipyard.harness.TestHarness.RocketConfig" ; "Rocket")]
    fn run(name: &str) -> Result<(), RippleIRErr> {
        equivalence_check_digitaltop(name)?;
        Ok(())
    }
}
