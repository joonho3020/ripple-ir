use crate::common::export_circuit;
use crate::passes::fir::from_ast::from_circuit;
use crate::passes::fir::remove_unnecessary_phi::remove_unnecessary_phi;
use crate::passes::fir::check_phi_nodes::check_phi_node_connections;
use crate::passes::fir::modify_names::add_sfx_to_module_names;
use crate::passes::fir::to_ast::to_ast;
use crate::common::RippleIRErr;
use crate::passes::ast::print::Printer;
use chirrtl_parser::parse_circuit;
use std::fs;
use std::fs::create_dir_all;
use std::path::Path;

fn remove_dir_if_exists(path: &str) -> Result<(), RippleIRErr> {
    let dir_path = Path::new(path);

    if dir_path.exists() && dir_path.is_dir() {
        fs::remove_dir_all(dir_path)?;
        println!("Removed directory: {}", path);
    }

    Ok(())
}

fn export(pfx: &str, firname: &str, circuit: &str) -> Result<(), RippleIRErr> {
    let outdir = format!("./test-outputs/{}/{}", firname, pfx);
    let firfile = format!("./test-outputs/{}/{}.{}.fir", firname, firname, pfx);

    // Perform cleanup
    remove_dir_if_exists(&outdir)?;
    create_dir_all(&outdir)?;

    // Write firrtl file
    std::fs::write(&firfile, &circuit)?;

    // Export verilog
    export_circuit(&firfile, &outdir)?;
    Ok(())
}

pub fn equivalence_check(input_fir: &str) -> Result<(), RippleIRErr> {
    let filename = format!("./test-inputs/{}.fir", input_fir);
    let source = std::fs::read_to_string(filename)?;
    export("golden", input_fir, &source)?;

    let circuit = parse_circuit(&source).expect("firrtl parser");
    let mut ir = from_circuit(&circuit);
    remove_unnecessary_phi(&mut ir);
    check_phi_node_connections(&ir)?;

// let old_hier = ir.hier.clone();
    add_sfx_to_module_names(&mut ir, "_impl");

    let circuit_reconstruct = to_ast(&ir);
    let mut printer = Printer::new();
    let circuit_str = printer.print_circuit(&circuit_reconstruct);
    export("impl", input_fir, &circuit_str)?;


    Ok(())
}

#[cfg(test)]
mod test {
    use test_case::test_case;
    use crate::common::RippleIRErr;
    use super::*;

    #[test_case("GCD" ; "GCD")]
    #[test_case("Hierarchy" ; "Hierarchy")]
    fn run(name: &str) -> Result<(), RippleIRErr> {
        equivalence_check(name)?;
        Ok(())
    }
}
