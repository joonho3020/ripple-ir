use ripple_ir::common::export_circuit;
use std::fs;

#[test]
fn test_export_gcd_circuit() {
    let fir_file = "test-inputs/GCD.fir";
    let out_dir = "tests";
    let out_file = format!("{}/GCD.sv", out_dir);

    // Ensure output directory exists
    fs::create_dir_all(out_dir).unwrap();

    // Remove output file if it exists
    let _ = fs::remove_file(&out_file);

    // Call export_circuit with the output file path
    export_circuit(fir_file, &out_file).expect("export_circuit should succeed");

    // Check that the output file was created
    assert!(fs::metadata(&out_file).is_ok(), "Output file was not created");
} 