#[cfg(test)]
mod test {
    use crate::passes::fir::from_ast::from_circuit_module;
    use crate::passes::fir::fir_simulator::{FirSimulator, FirValue};
    use chirrtl_parser::parse_circuit;
    use rusty_firrtl::Int;
    use std::fs;

    #[test]
    fn fir_simulator_adder() {
        let firrtl = fs::read_to_string("test-inputs/Adder.fir").expect("read Adder.fir");
        let circuit = parse_circuit(&firrtl).expect("parse FIRRTL");
        let (_name, fg) = from_circuit_module(&circuit.modules[0]);
        let mut sim = FirSimulator::new(fg);
        sim.set_input("a", Int::from(6u32));
        sim.set_input("b", Int::from(9u32));
        sim.run();
        sim.display();
        let output_val = sim.get_output("y");
        assert_eq!(output_val, Some(FirValue::Int(Int::from(15u32))), "Adder(6,9) should be 15");
    }
}