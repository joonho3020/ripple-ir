//iverilog -o gcd_sim GCD_tb.v GCD.sv/GCD.sv
//vvp gcd_sim

`timescale 1ns/1ps

module GCD_tb;
  reg clock = 0;
  reg reset = 0;
  reg [15:0] value1 = 0;
  reg [15:0] value2 = 0;
  reg loadingValues = 0;
  wire [15:0] outputGCD;
  wire outputValid;

  // Instantiate the GCD module
  GCD dut (
    .clock(clock),
    .reset(reset),
    .io_value1(value1),
    .io_value2(value2),
    .io_loadingValues(loadingValues),
    .io_outputGCD(outputGCD),
    .io_outputValid(outputValid)
  );

  // Clock generation
  always #5 clock = ~clock;

  initial begin
    $display("Starting GCD testbench");
    
    // Set input values before first clock edge
    value1 = 60;
    value2 = 48;
    loadingValues = 1;
    
    // Initial reset
    reset = 1;
    #10;
    reset = 0;
    #5; // Wait for first clock edge
    
    loadingValues = 0; // Deassert after first cycle
  end

  // Cycle counter
  reg [31:0] cycle_count = 0;

  always @(posedge clock) begin
    cycle_count = cycle_count + 1;
    $display("Cycle %11d: value1=%4d value2=%4d loadingValues=%2d x=%4d y=%4d outputGCD=%4d outputValid=%1b",
         cycle_count, value1, value2, loadingValues, dut.x, dut.y, outputGCD, outputValid);
    if (outputValid) begin
      $display("Final GCD: %d", outputGCD);
      $finish;
    end
    if (cycle_count >= 20) begin
      $display("Testbench timed out after 20 cycles without valid output.");
      $finish;
    end
  end
endmodule 