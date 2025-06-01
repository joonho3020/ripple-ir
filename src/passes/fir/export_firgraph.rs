#[cfg(test)]
mod test {
    use crate::common::RippleIRErr;
    use crate::passes::runner::run_passes_from_firrtl3_file;

    fn run(filepath: &str, outdir: &str, pfx: &str) -> Result<(), RippleIRErr> {
        let fir = run_passes_from_firrtl3_file(filepath)?;
// fir.export(outdir, pfx)?;
        Ok(())
    }

    #[test]
    fn queue_lo() -> Result<(), RippleIRErr> {
        run("./test-inputs-firrtl3/Queue.lo.fir", "./test-outputs", "firrtl3")
    }

    #[test]
    fn firesim_rocket() -> Result<(), RippleIRErr> {
        run("./test-inputs-firrtl3/FireSimRocket.fir", "./test-outputs", "firrtl3")
    }

    #[test]
    fn firesim_boom() -> Result<(), RippleIRErr> {
        run("./test-inputs-firrtl3/FireSimLargeBoom.fir", "./test-outputs", "firrtl3")
    }
}
