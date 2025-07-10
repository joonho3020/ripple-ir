#[cfg(test)]
mod test {
    use crate::passes::fir::from_ast::from_circuit;
    use crate::passes::fir::fir_simulator::{FirSimulator, FirValue};
    use crate::passes::runner::run_fir_passes;
    use chirrtl_parser::parse_circuit;
    use rusty_firrtl::Int;
    use std::fs;
    use crate::common::graphviz::GraphViz;
    use num_traits::ToPrimitive;
    use petgraph::visit::EdgeRef;

    #[test]
    fn fir_simulator_adder() {
        let firrtl = fs::read_to_string("test-inputs/Adder.fir").expect("read Adder.fir");
        let circuit = parse_circuit(&firrtl).expect("parse FIRRTL");
        
        let mut fir = from_circuit(&circuit);
        run_fir_passes(&mut fir).expect("run FIR passes");
        
        let (_name, fg) = fir.graphs.iter().next().unwrap();
        let fg = fg.clone();
        
        let mut sim = FirSimulator::new(fg);
        
        sim.set_bundle_input("io", "a", Int::from(3));
        sim.set_bundle_input("io", "b", Int::from(5));
        
        let a = sim.get_output("io.a");
        let b = sim.get_output("io.b");
        println!("Input a: {:?}", a);
        println!("Input b: {:?}", b);

        sim.run();
        let output_val = sim.get_output("io.c");
        println!("Output value: {:?}", output_val);
        sim.run();
        let output_val = sim.get_output("io.c");
        println!("Output value: {:?}", output_val);

        sim.display();
        sim.display_levelization();

        let output_val = sim.get_output("io.c");
    }

    #[test]
    fn fir_simulator_gcd() {
        let firrtl = fs::read_to_string("test-inputs/GCD.fir").expect("read GCD.fir");
        let circuit = parse_circuit(&firrtl).expect("parse FIRRTL");
        
        let mut fir = from_circuit(&circuit);
        run_fir_passes(&mut fir).expect("run FIR passes");

        let (_name, fg) = fir.graphs.iter().next().unwrap();
        let fg = fg.clone();

        let mut sim = crate::passes::fir::fir_simulator::FirSimulator::new(fg);

        // Set initial values
        sim.set_bundle_input("io", "value1", Int::from(60));
        sim.set_bundle_input("io", "value2", Int::from(48));
        sim.set_bundle_input("io", "loadingValues", Int::from(1));

        // Print input values
        let value1 = sim.get_output("io.value1");
        let value2 = sim.get_output("io.value2");
        println!("Input value1: {:?}", value1);
        println!("Input value2: {:?}", value2);

        // Run simulation cycles
        for cycle in 0..10 {
            sim.run();
            // Print register values for x and y after each cycle
            let x_val = sim.get_output("x");
            let y_val = sim.get_output("y");
            println!("Cycle {}: x = {:?}, y = {:?}", cycle, x_val, y_val);
            if cycle == 0 {
                sim.set_bundle_input("io", "loadingValues", Int::from(0));
            }
            let valid = sim.get_output("io.outputValid");
            if let Some(FirValue::Int(valid_val)) = valid {
                if valid_val.0.to_u64().unwrap_or(0) == 1 {
                    break;
                }
            }
        }

        let final_gcd = sim.get_output("io.outputGCD");
        let final_valid = sim.get_output("io.outputValid");
        println!("Final GCD: {:?}", final_gcd);
        println!("Final Valid: {:?}", final_valid);
        if let Some(FirValue::Int(gcd_val)) = final_gcd {
            let gcd_decimal = gcd_val.0.to_u64().unwrap_or(0);
            println!("Computed GCD: {}", gcd_decimal);
        }
        
        // Display graph structure
        sim.display();
        sim.display_levelization();
    }

}