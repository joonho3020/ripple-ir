#[cfg(test)]
mod test {
    use crate::passes::fir::from_ast::from_circuit;
    use crate::passes::fir::fir_simulator::{FirSimulator};
    use crate::passes::runner::run_fir_passes;
    use chirrtl_parser::parse_circuit;
    use rusty_firrtl::Int;
    use std::fs;
    use crate::common::graphviz::GraphViz;
    use num_traits::ToPrimitive;

    #[test]
    fn fir_simulator_adder() {
        let firrtl = fs::read_to_string("test-inputs/Adder.fir").expect("read Adder.fir");
        let circuit = parse_circuit(&firrtl).expect("parse FIRRTL");
        
        let mut fir = from_circuit(&circuit);
        run_fir_passes(&mut fir).expect("run FIR passes");
        
        let (_name, fg) = fir.graphs.iter().next().unwrap();
        let fg = fg.clone();
        
        println!("{}", fg.graphviz_string(None, None).unwrap());
        let mut sim = FirSimulator::new(fg);
        
        sim.set_bundle_input("io", "a", Int::from(3));
        sim.set_bundle_input("io", "b", Int::from(5));
        
        sim.run();
        let output_val = sim.get_output("io.c");
        println!("Output value: {:?}", output_val);
        sim.run();
        let output_val = sim.get_output("io.c");
        println!("Output value: {:?}", output_val);

        sim.display();
        sim.display_levelization();

        
        let output_val = sim.get_output("io.c");
        println!("Output value: {:?}", output_val);
    }

}