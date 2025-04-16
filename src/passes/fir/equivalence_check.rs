use crate::common::export_circuit;
use crate::passes::fir::from_ast::from_circuit;
use crate::passes::fir::remove_unnecessary_phi::remove_unnecessary_phi;
use crate::passes::fir::check_phi_nodes::check_phi_node_connections;
use crate::passes::fir::modify_names::add_sfx_to_module_names;
use crate::passes::fir::to_ast::to_ast;
use crate::common::RippleIRErr;
use crate::passes::ast::print::Printer;
use chirrtl_parser::ast::{CircuitModule, DefName, Identifier};
use chirrtl_parser::parse_circuit;
use indexmap::IndexSet;
use std::fs;
use std::fs::create_dir_all;
use std::path::Path;
use std::fs::File;
use std::io::Write;
use std::str::Lines;
use std::process::Command;
use spinoff::{Spinner, spinners};

pub fn equivalence_check(input_fir: &str) -> Result<(), RippleIRErr> {
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

    let top_name = old_hier.graph.node_weight(old_hier.root().unwrap()).unwrap().name();
    let mut top_sv_filename = verilog_outdir("golden", input_fir);
    if let Identifier::Name(x) = top_name {
        top_sv_filename.push_str(&format!("/{}.sv", x));
    }
    export_miter(&input_fir, &top_sv_filename)?;

    for cm in circuit.modules {
        if let CircuitModule::ExtModule(em) = cm.as_ref() {
            if let DefName(Identifier::Name(x)) = &em.defname {
                println!("exte module {:?}", x);
                let input = format!("./test-inputs/{}.v", x);
                copy_file(&input,
                     &format!("{}/{}.sv", verilog_outdir("golden", input_fir), x))?;
                copy_file(&input,
                     &format!("{}/{}.sv", verilog_outdir("impl", input_fir), x))?;
            }
        }
    }

    export_tcl(&input_fir)?;

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

    export_tcl(&input_fir)?;

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

fn remove_dir_if_exists(path: &str) -> Result<(), RippleIRErr> {
    let dir_path = Path::new(path);

    if dir_path.exists() && dir_path.is_dir() {
        fs::remove_dir_all(dir_path)?;
        println!("Removed directory: {}", path);
    }

    Ok(())
}

fn verilog_outdir(pfx: &str, firname: &str) -> String {
    format!("./test-outputs/{}/{}", firname, pfx)
}

fn firrtl_filename(pfx: &str, firname: &str) -> String {
    format!("./test-outputs/{}/{}.{}.fir", firname, firname, pfx)
}

fn miter_filename(firname: &str) -> String {
    format!("./test-outputs/{}/top_miter.v", firname)
}

fn tcl_filename(firname: &str) -> String {
    format!("./test-outputs/{}/check_equiv.tcl", firname)
}

fn export_firrtl_and_sv(pfx: &str, firname: &str, circuit: &str) -> Result<(), RippleIRErr> {
    let outdir = verilog_outdir(pfx, firname);
    let firfile = firrtl_filename(pfx, firname);

    // Perform cleanup
    remove_dir_if_exists(&outdir)?;
    create_dir_all(&outdir)?;

    // Write firrtl file
    std::fs::write(&firfile, &circuit)?;

    // Export verilog
    export_circuit(&firfile, &outdir)?;
    Ok(())
}

fn copy_file(src_path: &str, dst_path: &str) -> Result<(), RippleIRErr> {
    fs::copy(src_path, dst_path)?;
    Ok(())
}

// Struct to hold parsed module information
#[derive(Debug)]
struct Module {
    name: String,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
}

#[derive(Debug)]
struct Port {
    name: String,
    width: Option<String>,
}

impl Module {
    fn new(name: String) -> Self {
        Module {
            name,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

fn export_miter(firname: &str, top_sv_filename: &str) -> Result<(), RippleIRErr> {
    let top_module = std::fs::read_to_string(top_sv_filename)?;
    let module = parse_module(top_module.lines()).expect("Failed to parse module");

    let miter_content = generate_miter(&module);
    let mut file = File::create(&miter_filename(firname))?;
    file.write_all(miter_content.as_bytes())?;
    Ok(())
}

fn export_miter_digitaltop(firname: &str, top_sv_filename: &str) -> Result<(), RippleIRErr> {
    let top_module = std::fs::read_to_string(top_sv_filename)?;
    let module = parse_module(top_module.lines()).expect("Failed to parse module");

    let miter_content = generate_miter_digitaltop(&module);
    let mut file = File::create(&miter_filename(firname))?;
    file.write_all(miter_content.as_bytes())?;
    Ok(())
}

/// Parse SystemVerilog module from lines
fn parse_module(lines: Lines) -> Option<Module> {
    let mut module: Option<Module> = None;
    let mut in_module = false;

    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }

        // Detect module declaration
        if line.starts_with("module") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() > 1 {
                let name = parts[1].trim_end_matches('(').to_string();
                module = Some(Module::new(name));
                in_module = true;
            }
        }

        // Parse ports
        if in_module {
            if line.contains(");") {
                in_module = false;
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let direction = parts[0];
            let mut width: Option<String> = None;

            // Check for width specification
            let name = if parts.len() > 2 && parts[1].starts_with('[') {
                width = Some(parts[1].to_string());
                parts[2].trim_end_matches(',').to_string()
            } else {
                parts[1].trim_end_matches(',').to_string()
            };

            let port = Port { name, width };

            if let Some(ref mut m) = module {
                if direction == "input" {
                    m.inputs.push(port);
                } else if direction == "output" {
                    m.outputs.push(port);
                }
            }
        }
    }

    module
}

/// Generate miter module
fn generate_miter(module: &Module) -> String {
    let mut ret = String::new();

    // Module declaration
    ret.push_str("module top_miter(\n");
    ret.push_str("  input clock, reset");

    // Add other inputs
    for input in &module.inputs {
        if input.name != "clock" && input.name != "reset" {
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
    ret.push_str("    .clock(clock),\n");
    ret.push_str("    .reset(reset),\n");
    let has_oport = !module.outputs.is_empty();
    let iport_cnt = module.inputs.iter().count();
    for (i, input) in module.inputs.iter().enumerate() {
        if input.name != "clock" && input.name != "reset" {
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
    ret.push_str("    .clock(clock),\n");
    ret.push_str("    .reset(reset),\n");
    for (i, input) in module.inputs.iter().enumerate() {
        if input.name != "clock" && input.name != "reset" {
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


fn export_tcl(firname: &str) -> Result<(), RippleIRErr> {
    let golden_dir = verilog_outdir("golden", firname);
    let impl_dir = verilog_outdir("impl", firname);
    let output_file = tcl_filename(firname);

    // Collect .sv files from golden
    let golden_files: Vec<String> = fs::read_dir(golden_dir)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|s| s.to_str()) == Some("sv") {
                Some(path.to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect();

    // Collect .sv files from impl
    let impl_files: Vec<String> = fs::read_dir(impl_dir)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|s| s.to_str()) == Some("sv") {
                Some(path.to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect();

    // Create the TCL script content
    let mut tcl_content = String::new();
    tcl_content.push_str("# Clear previous session\n");
    tcl_content.push_str("clear -all\n\n");
    tcl_content.push_str("# Read the designs\n");

    // Add golden files
    for file in &golden_files {
        tcl_content.push_str(&format!("analyze -sv {}\n", file));
    }

    // Add impl files
    for file in &impl_files {
        tcl_content.push_str(&format!("analyze -sv {}\n", file));
    }

    // Add top_miter.v and remaining commands
    tcl_content.push_str(&format!("analyze -sv {}\n\n", miter_filename(firname)));
    tcl_content.push_str("# Elaborate designs\n");
    tcl_content.push_str("elaborate -top top_miter\n\n");
    tcl_content.push_str("# Set clock and reset signals\n");
    tcl_content.push_str("clock clock\n");
    tcl_content.push_str("reset reset\n\n");
    tcl_content.push_str("assert -disable {^top_miter\\.(ref_inst|impl_inst)\\..*} -regexp\n");
    tcl_content.push_str("prove -all\n\n");
    tcl_content.push_str("report\n");
    tcl_content.push_str("exit\n");

    // Write to check_equiv.txt
    let mut file = fs::File::create(output_file)?;
    file.write_all(tcl_content.as_bytes())?;

    Ok(())
}

fn run_jaspergold<P: AsRef<Path>>(firname: &str, dir: P) -> Result<EquivStatus, RippleIRErr> {
    let mut spinner = Spinner::new(
        spinners::Dots,
        format!("Running JasperGold on {}...", firname),
        None);

    let status = Command::new("jaspergold")
        .arg("-allow_unsupported_OS")
        .arg("-no_gui")
        .arg("-tcl")
        .arg(tcl_filename(firname))
        .current_dir(dir)
        .output()
        .expect("jaspergold to run");

    spinner.success("Finished running JasperGold");

    let stdout = String::from_utf8(status.stdout)?;
    let jasper_status = parse_jasper_summary(&stdout);

    Ok(jasper_status)
}

pub enum EquivStatus {
    Proven,
    CounterExample,
    NothingToProve,
    Unknown,
}

pub fn parse_jasper_summary(stdout: &str) -> EquivStatus {
    let mut in_summary = false;
    let mut proven = 0;
    let mut cex = 0;

    for line in stdout.lines() {
        if line.contains("SUMMARY") {
            in_summary = true;
            continue;
        }

        if in_summary {
            // exit summary if we hit another empty separator or unrelated line
            if line.trim().is_empty() {
                in_summary = false;
                continue;
            }

            let tokens: Vec<_> = line.split(':').map(str::trim).collect();
            if tokens.len() != 2 {
                continue;
            }

            match tokens[0] {
                "- proven" => {
                    proven = tokens[1].split_whitespace().next().unwrap_or("0").parse().unwrap_or(0);
                }
                "- cex" | "- ar_cex" => {
                    cex += tokens[1].split_whitespace().next().unwrap_or("0").parse::<u32>().unwrap_or(0);
                }
                _ => {}
            }
        }
    }

// println!("proven {} cex {}", proven, cex);

    if cex > 0 {
        EquivStatus::CounterExample
    } else if proven > 0 {
        EquivStatus::Proven
    } else if cex == 0 && proven == 0 {
        EquivStatus::NothingToProve
    } else {
        EquivStatus::Unknown
    }
}

#[cfg(test)]
mod test {
    use test_case::test_case;
    use crate::common::RippleIRErr;
    use super::*;

    #[test_case("GCD" ; "GCD")]
    #[test_case("Hierarchy" ; "Hierarchy")]
    #[test_case("NestedWhen" ; "NestedWhen")]
    #[test_case("LCS1" ; "LCS1")]
    #[test_case("LCS2" ; "LCS2")]
    #[test_case("LCS3" ; "LCS3")]
    #[test_case("LCS4" ; "LCS4")]
    #[test_case("LCS5" ; "LCS5")]
    #[test_case("LCS6" ; "LCS6")]
    #[test_case("LCS7" ; "LCS7")]
    #[test_case("LCS8" ; "LCS8")]
    #[test_case("BitSel1" ; "BitSel1")]
    #[test_case("BitSel2" ; "BitSel2")]
    #[test_case("RegInit" ; "RegInit")]
    #[test_case("RegInitWire" ; "RegInitWire")]
    #[test_case("SinglePortSRAM" ; "SinglePortSRAM")]
    #[test_case("OneReadOneWritePortSRAM" ; "OneReadOneWritePortSRAM")]
    #[test_case("OneReadOneReadWritePortSRAM" ; "OneReadOneReadWritePortSRAM")]
    #[test_case("MSHR" ; "MSHR")]
    #[test_case("TLBundleQueue" ; "TLBundleQueue")]
    #[test_case("ListBuffer" ; "ListBuffer")]
    #[test_case("Atomics" ; "Atomics")]
    #[test_case("PhitArbiter" ; "PhitArbiter")]
    #[test_case("TLMonitor" ; "TLMonitor")]
    #[test_case("TLBusBypassBar" ; "TLBusBypassBar")]
    #[test_case("WireRegInsideWhen" ; "WireRegInsideWhen")]
    #[test_case("MultiWhen" ; "MultiWhen")]
    #[test_case("DCacheDataArray" ; "DCacheDataArray")]
    fn run(name: &str) -> Result<(), RippleIRErr> {
        equivalence_check(name)?;
        Ok(())
    }

    #[test_case("chipyard.harness.TestHarness.RocketConfig" ; "Rocket")]
    fn run_digitaltop(name: &str) -> Result<(), RippleIRErr> {
        equivalence_check_digitaltop(name)?;
        Ok(())
    }
}
