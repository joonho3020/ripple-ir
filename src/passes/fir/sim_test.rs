#[cfg(test)]
mod test {
    use crate::passes::fir::from_ast::from_circuit;
    use crate::passes::fir::fir_simulator::{FirSimulator, FirValue};
    use crate::passes::runner::run_fir_passes;
    use chirrtl_parser::parse_circuit;
    use rusty_firrtl::Int;
    use std::fs;
    use num_traits::ToPrimitive;

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

        sim.set_bundle_input("io", "value1", Int::from(60));
        sim.set_bundle_input("io", "value2", Int::from(48));
        sim.set_bundle_input("io", "loadingValues", Int::from(1));

        sim.run();
        println!("Cycle          1: value1={:?} value2={:?} loadingValues={:?} x={:?} y={:?} outputGCD={:?} outputValid={:?}",
            sim.get_output("io.value1"),
            sim.get_output("io.value2"),
            sim.get_output("io.loadingValues"),
            sim.get_output("x"),
            sim.get_output("y"),
            sim.get_output("io.outputGCD"),
            sim.get_output("io.outputValid"));

        for cycle in 2..12 {
            if cycle == 2 {
                sim.set_bundle_input("io", "loadingValues", Int::from(0));
            }
            sim.run();
            println!("Cycle         {:2}: value1={:?} value2={:?} loadingValues={:?} x={:?} y={:?} outputGCD={:?} outputValid={:?}",
                cycle,
                sim.get_output("io.value1"),
                sim.get_output("io.value2"),
                sim.get_output("io.loadingValues"),
                sim.get_output("x"),
                sim.get_output("y"),
                sim.get_output("io.outputGCD"),
                sim.get_output("io.outputValid"));
            // Break if io.outputValid is 1
            if let Some(FirValue::Int(i)) = sim.get_output("io.outputValid") {
                if i.0.to_u64().unwrap_or(0) == 1 {
                    break;
                }
            }
        }

        let final_gcd = sim.get_output("io.outputGCD");
        println!("Final GCD: {:?}", final_gcd);

        sim.display();
        sim.display_levelization();
    }

    #[test]
    fn fir_simulator_fir() {
        let firrtl = fs::read_to_string("test-inputs/Fir.fir").expect("read Fir.fir");
        let circuit = parse_circuit(&firrtl).expect("parse FIRRTL");
        
        let mut fir = from_circuit(&circuit);
        run_fir_passes(&mut fir).expect("run FIR passes");

        let (_name, fg) = fir.graphs.iter().next().unwrap();
        let fg = fg.clone();

        // Print the FIR graph structure
        println!("FIR Graph: {:#?}", fg);

        let mut sim = FirSimulator::new(fg);

        sim.set_bundle_input("io", "consts[0]", Int::from(1));
        sim.set_bundle_input("io", "consts[1]", Int::from(1));
        sim.set_bundle_input("io", "consts[2]", Int::from(1));
        sim.set_bundle_input("io", "consts[3]", Int::from(1));

        // Assert reset for the first cycle
        sim.set_input("reset", Int::from(1));
        sim.run();
        println!("Cycle   1 (reset=1): input={:?} valid={:?} taps=[{:?},{:?},{:?}] output={:?}",
            sim.get_output("io.in"),
            sim.get_output("io.valid"),
            sim.get_output("taps_1"),
            sim.get_output("taps_2"),
            sim.get_output("taps_3"),
            sim.get_output("io.out"));

        // Deassert reset for normal operation
        sim.set_input("reset", Int::from(0));

        let test_inputs = [2, 4, 6, 8, 10, 12];
        
        for (cycle, &input_val) in test_inputs.iter().enumerate() {
            let cycle_num = cycle + 2;
            
            // Set input and valid
            sim.set_bundle_input("io", "in", Int::from(input_val));
            sim.set_bundle_input("io", "valid", Int::from(1));
            // reset remains deasserted
            sim.set_input("reset", Int::from(0));
            sim.run();
            println!("Cycle  {:2} (reset=0): input={:?} valid={:?} taps=[{:?},{:?},{:?}] output={:?}",
                cycle_num,
                sim.get_output("io.in"),
                sim.get_output("io.valid"),
                sim.get_output("taps_1"),
                sim.get_output("taps_2"),
                sim.get_output("taps_3"),
                sim.get_output("io.out"));
        }

        // Stop sending data (valid=0) and let filter settle
        sim.set_bundle_input("io", "valid", Int::from(0));
        for cycle in 8..11 {
            sim.set_input("reset", Int::from(0));
            sim.run();
            println!("Cycle  {:2} (reset=0): input={:?} valid={:?} taps=[{:?},{:?},{:?}] output={:?}",
                cycle,
                sim.get_output("io.in"),
                sim.get_output("io.valid"),
                sim.get_output("taps_1"),
                sim.get_output("taps_2"),
                sim.get_output("taps_3"),
                sim.get_output("io.out"));
        }

        println!("Test completed");
        sim.display();
        sim.display_levelization();
    }

}