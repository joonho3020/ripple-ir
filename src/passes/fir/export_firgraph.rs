#[cfg(test)]
mod test {
    use crate::common::RippleIRErr;
    use crate::passes::ast::print::*;
    use crate::passes::ast::firrtl3_print::FIRRTL3Printer;
    use crate::passes::fir::to_ast_firrtl3::to_ast_firrtl3;
    use crate::passes::runner::run_passes_from_firrtl3_file;

    fn run(name: &str, outdir: &str, pfx: &str) -> Result<(), RippleIRErr> {
        let filepath = format!("./test-inputs-firrtl3/{}.fir", name);
        let fir = run_passes_from_firrtl3_file(&filepath)?;

        let circuit_reconstruct = to_ast_firrtl3(&fir);

        let mut printer = FIRRTL3Printer::new();
        let circuit_reconstruct_str = printer.print_circuit(&circuit_reconstruct);
        let firrtl = format!("./{}/{}.{}.fir", outdir, name, pfx);
        std::fs::write(&firrtl, circuit_reconstruct_str)?;

// fir.export(outdir, pfx)?;
        Ok(())
    }

    #[test]
    fn queue_lo() -> Result<(), RippleIRErr> {
        run("Queue.lo", "./test-outputs", "firrtl3")
    }

    #[test]
    fn firesim_rocket() -> Result<(), RippleIRErr> {
        run("FireSimRocket", "./test-outputs", "firrtl3")
    }

    #[test]
    fn firesim_boom() -> Result<(), RippleIRErr> {
        run("FireSimLargeBoom", "./test-outputs", "firrtl3")
    }
}
