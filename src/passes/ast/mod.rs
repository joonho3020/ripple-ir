pub mod check_ast_assumption;
pub mod print;
pub mod gumtree;
pub mod firrtlgraph;
pub mod gumtree_graphviz;

#[cfg(test)]
mod test {
    use chirrtl_parser::parse_circuit;
    use crate::common::RippleIRErr;
    use crate::passes::ast::check_ast_assumption::check_ast_assumption;

    fn run_check_assumption(input: &str) -> Result<(), RippleIRErr> {
        let source = std::fs::read_to_string(input)?;
        let circuit = parse_circuit(&source).expect("firrtl parser");
        check_ast_assumption(&circuit);
        Ok(())
    }

    #[test]
    fn rocket_ast_assumption() {
        run_check_assumption("./test-inputs/chipyard.harness.TestHarness.RocketConfig.fir")
            .expect("rocket ast assumption");
    }

    #[test]
    fn boom_ast_assumption() {
        run_check_assumption("./test-inputs/chipyard.harness.TestHarness.LargeBoomV3Config.fir")
            .expect("boom ast assumption");
    }
}
