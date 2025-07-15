//iverilog -o fir_sim Fir_tb.v Fir.sv/Fir.sv
//vvp fir_sim

`timescale 1ns/1ps

module Fir_tb;
  reg clock = 0;
  reg reset = 0;
  reg [3:0] input_data = 0;
  reg input_valid = 0;
  reg [3:0] consts_0 = 0;
  reg [3:0] consts_1 = 0;
  reg [3:0] consts_2 = 0;
  reg [3:0] consts_3 = 0;
  wire [3:0] output_data;

  // Instantiate the FIR module
  Fir dut (
    .clock(clock),
    .reset(reset),
    .io_in(input_data),
    .io_valid(input_valid),
    .io_out(output_data),
    .io_consts_0(consts_0),
    .io_consts_1(consts_1),
    .io_consts_2(consts_2),
    .io_consts_3(consts_3)
  );

  // Clock generation
  always #5 clock = ~clock;

  initial begin
    $display("Starting FIR filter testbench");
    
    // Set FIR coefficients (example: simple averaging filter)
    consts_0 = 4'd1;  // 1/4 weight for current input
    consts_1 = 4'd1;  // 1/4 weight for tap_1
    consts_2 = 4'd1;  // 1/4 weight for tap_2
    consts_3 = 4'd1;  // 1/4 weight for tap_3
    
    // Initial reset
    reset = 1;
    #10;
    reset = 0;
    #5; // Wait for first clock edge
    
    // Test sequence: send input values
    input_valid = 1;
    input_data = 4'd2;  // First input
    #10;
    
    input_data = 4'd4;  // Second input
    #10;
    
    input_data = 4'd6;  // Third input
    #10;
    
    input_data = 4'd8;  // Fourth input
    #10;
    
    input_data = 4'd10; // Fifth input
    #10;
    
    input_data = 4'd12; // Sixth input
    #10;
    
    input_valid = 0;    // Stop sending data
    #20;                // Let filter settle
    
    $display("Test completed");
    $finish;
  end

  // Cycle counter
  reg [31:0] cycle_count = 0;

  always @(posedge clock) begin
    cycle_count = cycle_count + 1;
    $display("Cycle %3d: input=%2d valid=%1b taps=[%2d,%2d,%2d] output=%2d",
         cycle_count, input_data, input_valid, dut.taps_1, dut.taps_2, dut.taps_3, output_data);
    
    if (cycle_count >= 20) begin
      $display("Testbench completed after 20 cycles.");
      $finish;
    end
  end
endmodule 