mod lexer;
mod ast;
use lalrpop_util::lalrpop_mod;
lalrpop_mod!(pub firrtl);



pub type Int = i128;





#[cfg(test)]
mod lexer_test {
    use crate::lexer::*;

    fn run(source: &str) {
        let mut lex = FIRRTLLexer::new(source);
        while let Some(ts) = lex.next_token() {
            println!("{:?}", ts);
            match ts.token {
                Token::Error => {
                    println!("{:?}", ts);
                    panic!("Got a error token");
                }
                _ => { }
            }
        }
    }

    #[test]
    fn gcd() {
        let source =
r#"FIRRTL version 3.3.0
circuit GCD :
  module GCD : @[src/main/scala/gcd/GCD.scala 15:7]
    input clock : Clock @[src/main/scala/gcd/GCD.scala 15:7]
    input reset : UInt<1> @[src/main/scala/gcd/GCD.scala 15:7]
    output io : { flip value1 : UInt<16>, flip value2 : UInt<16>, flip loadingValues : UInt<1>, outputGCD : UInt<16>, outputValid : UInt<1>} @[src/main/scala/gcd/GCD.scala 16:14]

    reg x : UInt, clock @[src/main/scala/gcd/GCD.scala 24:15]
    reg y : UInt, clock @[src/main/scala/gcd/GCD.scala 25:15]
    node _T = gt(x, y) @[src/main/scala/gcd/GCD.scala 27:10]
    when _T : @[src/main/scala/gcd/GCD.scala 27:15]
      node _x_T = sub(x, y) @[src/main/scala/gcd/GCD.scala 27:24]
      node _x_T_1 = tail(_x_T, 1) @[src/main/scala/gcd/GCD.scala 27:24]
      connect x, _x_T_1 @[src/main/scala/gcd/GCD.scala 27:19]
    else :
      node _y_T = sub(y, x) @[src/main/scala/gcd/GCD.scala 28:25]
      node _y_T_1 = tail(_y_T, 1) @[src/main/scala/gcd/GCD.scala 28:25]
      connect y, _y_T_1 @[src/main/scala/gcd/GCD.scala 28:20]
    when io.loadingValues : @[src/main/scala/gcd/GCD.scala 30:26]
      connect x, io.value1 @[src/main/scala/gcd/GCD.scala 31:7]
      connect y, io.value2 @[src/main/scala/gcd/GCD.scala 32:7]
    connect io.outputGCD, x @[src/main/scala/gcd/GCD.scala 35:16]
    node _io_outputValid_T = eq(y, UInt<1>(0h0)) @[src/main/scala/gcd/GCD.scala 36:23]
    connect io.outputValid, _io_outputValid_T @[src/main/scala/gcd/GCD.scala 36:18]
"#;

        run(source);
    }

    #[test]
    fn one_read_one_write_sram() {
        let source =
r#"FIRRTL version 3.3.0
circuit OneReadOneWritePortSRAM :
  module OneReadOneWritePortSRAM : @[src/main/scala/gcd/SRAM.scala 10:7]
    input clock : Clock @[src/main/scala/gcd/SRAM.scala 10:7]
    input reset : UInt<1> @[src/main/scala/gcd/SRAM.scala 10:7]
    output io : { flip ren : UInt<1>, flip raddr : UInt<3>, rdata : UInt<2>[4], flip wen : UInt<1>, flip waddr : UInt<3>, flip wdata : UInt<2>[4], flip wmask : UInt<1>[4]} @[src/main/scala/gcd/SRAM.scala 11:14]

    smem mem : UInt<2>[4] [8] @[src/main/scala/gcd/SRAM.scala 22:24]
    when io.wen : @[src/main/scala/gcd/SRAM.scala 23:17]
      write mport MPORT = mem[io.waddr], clock @[src/main/scala/gcd/SRAM.scala 24:14]
      when io.wmask[0] : @[src/main/scala/gcd/SRAM.scala 24:14]
        connect MPORT[0], io.wdata[0] @[src/main/scala/gcd/SRAM.scala 24:14]
      when io.wmask[1] : @[src/main/scala/gcd/SRAM.scala 24:14]
        connect MPORT[1], io.wdata[1] @[src/main/scala/gcd/SRAM.scala 24:14]
      when io.wmask[2] : @[src/main/scala/gcd/SRAM.scala 24:14]
        connect MPORT[2], io.wdata[2] @[src/main/scala/gcd/SRAM.scala 24:14]
      when io.wmask[3] : @[src/main/scala/gcd/SRAM.scala 24:14]
        connect MPORT[3], io.wdata[3] @[src/main/scala/gcd/SRAM.scala 24:14]
    wire _WIRE : UInt<3> @[src/main/scala/gcd/SRAM.scala 26:23]
    invalidate _WIRE @[src/main/scala/gcd/SRAM.scala 26:23]
    when io.ren : @[src/main/scala/gcd/SRAM.scala 26:23]
      connect _WIRE, io.raddr @[src/main/scala/gcd/SRAM.scala 26:23]
      read mport MPORT_1 = mem[_WIRE], clock @[src/main/scala/gcd/SRAM.scala 26:23]
    connect io.rdata, MPORT_1 @[src/main/scala/gcd/SRAM.scala 26:12]
    "#;

        run(source);
    }

    #[test]
    fn ports_2() {
        let source = r#"output io : { flip a : UInt<2>, flip b : UInt<2>, flip c : UInt<2>, flip sel : UInt<2>, output : UInt<2>}"#;
        run(source);
    }

    #[test]
    fn extmodule() {
        let source =
r#"extmodule plusarg_reader : @[generators/rocket-chip/src/main/scala/util/PlusArg.scala 45:7]
     output out : UInt<32>
     defname = plusarg_reader
     parameter DEFAULT = 0
     parameter FORMAT = "tilelink_timeout=%d"
     parameter WIDTH = 32"#;
        run(source);
    }

    #[test]
    fn stmts() {
        let source =
r#"reg x : UInt, clock @[src/main/scala/gcd/GCD.scala 24:15]
reg y : UInt, clock @[src/main/scala/gcd/GCD.scala 25:15]
node _T = gt(x, y) @[src/main/scala/gcd/GCD.scala 27:10]
when _T : @[src/main/scala/gcd/GCD.scala 27:15]
  node _x_T = sub(x, y) @[src/main/scala/gcd/GCD.scala 27:24]
  node _x_T_1 = tail(_x_T, 1) @[src/main/scala/gcd/GCD.scala 27:24]
  connect x, _x_T_1 @[src/main/scala/gcd/GCD.scala 27:19]
else :
  node _y_T = sub(y, x) @[src/main/scala/gcd/GCD.scala 28:25]
  node _y_T_1 = tail(_y_T, 1) @[src/main/scala/gcd/GCD.scala 28:25]
  connect y, _y_T_1 @[src/main/scala/gcd/GCD.scala 28:20]
when io.loadingValues : @[src/main/scala/gcd/GCD.scala 30:26]
  connect x, io.value1 @[src/main/scala/gcd/GCD.scala 31:7]
  connect y, io.value2 @[src/main/scala/gcd/GCD.scala 32:7]
"#;
        run(source);
    }

    #[test]
    fn circuit_annos() {
        let source =
r#"FIRRTL version 3.3.0
circuit GCD :%[[
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~TestHarness|IntXbar_i1_o1",
    "group":"IntXbar_i1_o1"
  }
]]
  module GCD : @[src/main/scala/gcd/GCD.scala 15:7]
    input clock : Clock @[src/main/scala/gcd/GCD.scala 15:7]
    input reset : UInt<1> @[src/main/scala/gcd/GCD.scala 15:7]
    output io : { flip value1 : UInt<16>, flip value2 : UInt<16>, flip loadingValues : UInt<1>, outputGCD : UInt<16>, outputValid : UInt<1>} @[src/main/scala/gcd/GCD.scala 16:14]

    reg x : UInt, clock @[src/main/scala/gcd/GCD.scala 24:15]
    reg y : UInt, clock @[src/main/scala/gcd/GCD.scala 25:15]
    node _T = gt(x, y) @[src/main/scala/gcd/GCD.scala 27:10]
    when _T : @[src/main/scala/gcd/GCD.scala 27:15]
      node _x_T = sub(x, y) @[src/main/scala/gcd/GCD.scala 27:24]
      node _x_T_1 = tail(_x_T, 1) @[src/main/scala/gcd/GCD.scala 27:24]
      connect x, _x_T_1 @[src/main/scala/gcd/GCD.scala 27:19]
    else :
      node _y_T = sub(y, x) @[src/main/scala/gcd/GCD.scala 28:25]
      node _y_T_1 = tail(_y_T, 1) @[src/main/scala/gcd/GCD.scala 28:25]
      connect y, _y_T_1 @[src/main/scala/gcd/GCD.scala 28:20]
    when io.loadingValues : @[src/main/scala/gcd/GCD.scala 30:26]
      connect x, io.value1 @[src/main/scala/gcd/GCD.scala 31:7]
      connect y, io.value2 @[src/main/scala/gcd/GCD.scala 32:7]
    connect io.outputGCD, x @[src/main/scala/gcd/GCD.scala 35:16]
    node _io_outputValid_T = eq(y, UInt<1>(0h0)) @[src/main/scala/gcd/GCD.scala 36:23]
    connect io.outputValid, _io_outputValid_T @[src/main/scala/gcd/GCD.scala 36:18]
"#;
        run(&source);
    }

    #[test]
    fn rocketconfig() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./test-inputs/chipyard.harness.TestHarness.RocketConfig.fir")?;
        run(&source);
        Ok(())
    }

    #[test]
    fn largeboomconfig() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./test-inputs/chipyard.harness.TestHarness.LargeBoomV3Config.fir")?;
        run(&source);
        Ok(())
    }

}



#[cfg(test)]
mod parser_test {
    use crate::lexer::*;
    use crate::firrtl::*;

    #[test]
    fn stmts() {
        let source =
r#"reg x : UInt, clock @[src/main/scala/gcd/GCD.scala 24:15]
reg y : UInt, clock @[src/main/scala/gcd/GCD.scala 25:15]
node _T = gt(x, y) @[src/main/scala/gcd/GCD.scala 27:10]
when _T : @[src/main/scala/gcd/GCD.scala 27:15]
  node _x_T = sub(x, y) @[src/main/scala/gcd/GCD.scala 27:24]
  node _x_T_1 = tail(_x_T, 1) @[src/main/scala/gcd/GCD.scala 27:24]
  connect x, _x_T_1 @[src/main/scala/gcd/GCD.scala 27:19]
else :
  node _y_T = sub(y, x) @[src/main/scala/gcd/GCD.scala 28:25]
  node _y_T_1 = tail(_y_T, 1) @[src/main/scala/gcd/GCD.scala 28:25]
  connect y, _y_T_1 @[src/main/scala/gcd/GCD.scala 28:20]
when io.loadingValues : @[src/main/scala/gcd/GCD.scala 30:26]
  connect x, io.value1 @[src/main/scala/gcd/GCD.scala 31:7]
  connect y, io.value2 @[src/main/scala/gcd/GCD.scala 32:7]
"#;
        let lexer = FIRRTLLexer::new(source);
        let parser = StmtsParser::new();
        let ast = parser.parse(lexer).unwrap();

        for stmt in ast.iter() {
            stmt.traverse();
        }
    }

    #[test]
    fn ports() {
        let source =
r#"
input clock : Clock @[src/main/scala/gcd/GCD.scala 15:7]
input reset : UInt<1> @[src/main/scala/gcd/GCD.scala 15:7]
output io : { flip value1 : UInt<16>, flip value2 : UInt<16>, flip loadingValues : UInt<1>, outputGCD : UInt<16>, outputValid : UInt<1>} @[src/main/scala/gcd/GCD.scala 16:14]
"#;
        let lexer = FIRRTLLexer::new(source);
        let parser = PortsParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);
    }

    #[test]
    fn ports_2() {
        let source = r#"output io : { flip a : UInt<2>, flip b : UInt<2>, flip c : UInt<2>, flip sel : UInt<2>, output : UInt<2>}"#;
        let lexer = FIRRTLLexer::new(source);
        let parser = PortsParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);
    }

    #[test]
    fn ports_3() {
        let source = r#"output io : { flip in : { a : { ready : UInt<1>, valid : UInt<1>, bits : { opcode : UInt<3>, param : UInt<3>, size : UInt<4>, source : UInt<5>, address : UInt<32>, user : { }, echo : { }, mask : UInt<8>, data : UInt<64>, corrupt : UInt<1>}}, d : { ready : UInt<1>, valid : UInt<1>, bits : { opcode : UInt<3>, param : UInt<2>, size : UInt<4>, source : UInt<5>, sink : UInt<3>, denied : UInt<1>, user : { }, echo : { }, data : UInt<64>, corrupt : UInt<1>}}}} @[generators/rocket-chip/src/main/scala/tilelink/Monitor.scala 20:14]"#;
        let lexer = FIRRTLLexer::new(source);
        let parser = PortsParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);
    }

    #[test]
    fn when() {
        let source =
r#"
when io.in.a.valid : @[generators/rocket-chip/src/main/scala/tilelink/Monitor.scala 372:27]
  node _T = leq(io.in.a.bits.opcode, UInt<3>(0h7)) @[generators/rocket-chip/src/main/scala/tilelink/Bundles.scala 42:24]
  node _T_1 = asUInt(reset) @[generators/rocket-chip/src/main/scala/tilelink/Monitor.scala 45:11]
  node _T_2 = eq(_T_1, UInt<1>(0h0)) @[generators/rocket-chip/src/main/scala/tilelink/Monitor.scala 45:11]
  when _T_2 : @[generators/rocket-chip/src/main/scala/tilelink/Monitor.scala 45:11]
    node _T_3 = eq(_T, UInt<1>(0h0)) @[generators/rocket-chip/src/main/scala/tilelink/Monitor.scala 45:11]
    when _T_3 : @[generators/rocket-chip/src/main/scala/tilelink/Monitor.scala 45:11]
      printf(clock, UInt<1>(0h1), "Assertion failed: 'A' channel has invalid opcode (connected at generators/rocket-chip/src/main/scala/subsystem/SystemBus.scala:48:55)\n    at Monitor.scala:45 assert(cond, message)\n") : printf @[generators/rocket-chip/src/main/scala/tilelink/Monitor.scala 45:11]
    assert(clock, _T, UInt<1>(0h1), "") : assert @[generators/rocket-chip/src/main/scala/tilelink/Monitor.scala 45:11]
"#;

        let lexer = FIRRTLLexer::new(source);
        let parser = StmtsParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);

    }

    #[test]
    fn module() {
        let source =
r#"
module GCD : @[src/main/scala/gcd/GCD.scala 15:7]
  input clock : Clock @[src/main/scala/gcd/GCD.scala 15:7]
  input reset : UInt<1> @[src/main/scala/gcd/GCD.scala 15:7]
  output io : { flip value1 : UInt<16>, flip value2 : UInt<16>, flip loadingValues : UInt<1>, outputGCD : UInt<16>, outputValid : UInt<1>} @[src/main/scala/gcd/GCD.scala 16:14]

  reg x : UInt, clock @[src/main/scala/gcd/GCD.scala 24:15]
  reg y : UInt, clock @[src/main/scala/gcd/GCD.scala 25:15]
  node _T = gt(x, y) @[src/main/scala/gcd/GCD.scala 27:10]
  when _T : @[src/main/scala/gcd/GCD.scala 27:15]
    node _x_T = sub(x, y) @[src/main/scala/gcd/GCD.scala 27:24]
    node _x_T_1 = tail(_x_T, 1) @[src/main/scala/gcd/GCD.scala 27:24]
    connect x, _x_T_1 @[src/main/scala/gcd/GCD.scala 27:19]
  else :
    node _y_T = sub(y, x) @[src/main/scala/gcd/GCD.scala 28:25]
    node _y_T_1 = tail(_y_T, 1) @[src/main/scala/gcd/GCD.scala 28:25]
    connect y, _y_T_1 @[src/main/scala/gcd/GCD.scala 28:20]
  when io.loadingValues : @[src/main/scala/gcd/GCD.scala 30:26]
    connect x, io.value1 @[src/main/scala/gcd/GCD.scala 31:7]
    connect y, io.value2 @[src/main/scala/gcd/GCD.scala 32:7]
  connect io.outputGCD, x @[src/main/scala/gcd/GCD.scala 35:16]
  node _io_outputValid_T = eq(y, UInt<1>(0h0)) @[src/main/scala/gcd/GCD.scala 36:23]
  connect io.outputValid, _io_outputValid_T @[src/main/scala/gcd/GCD.scala 36:18]
"#;
        let lexer = FIRRTLLexer::new(source);
        let parser = ModuleParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);
    }
    #[test]
    fn version() {
        let source = r#"FIRRTL version 3.3.0"#;
        let lexer = FIRRTLLexer::new(source);
        let parser = VersionParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);
    }

    #[test]
    fn circuit() {
        let source =
r#"FIRRTL version 3.3.0
circuit GCD :
  module GCD : @[src/main/scala/gcd/GCD.scala 15:7]
    input clock : Clock @[src/main/scala/gcd/GCD.scala 15:7]
    input reset : UInt<1> @[src/main/scala/gcd/GCD.scala 15:7]
    output io : { flip value1 : UInt<16>, flip value2 : UInt<16>, flip loadingValues : UInt<1>, outputGCD : UInt<16>, outputValid : UInt<1>} @[src/main/scala/gcd/GCD.scala 16:14]

    reg x : UInt, clock @[src/main/scala/gcd/GCD.scala 24:15]
    reg y : UInt, clock @[src/main/scala/gcd/GCD.scala 25:15]
    node _T = gt(x, y) @[src/main/scala/gcd/GCD.scala 27:10]
    when _T : @[src/main/scala/gcd/GCD.scala 27:15]
      node _x_T = sub(x, y) @[src/main/scala/gcd/GCD.scala 27:24]
      node _x_T_1 = tail(_x_T, 1) @[src/main/scala/gcd/GCD.scala 27:24]
      connect x, _x_T_1 @[src/main/scala/gcd/GCD.scala 27:19]
    else :
      node _y_T = sub(y, x) @[src/main/scala/gcd/GCD.scala 28:25]
      node _y_T_1 = tail(_y_T, 1) @[src/main/scala/gcd/GCD.scala 28:25]
      connect y, _y_T_1 @[src/main/scala/gcd/GCD.scala 28:20]
    when io.loadingValues : @[src/main/scala/gcd/GCD.scala 30:26]
      connect x, io.value1 @[src/main/scala/gcd/GCD.scala 31:7]
      connect y, io.value2 @[src/main/scala/gcd/GCD.scala 32:7]
    connect io.outputGCD, x @[src/main/scala/gcd/GCD.scala 35:16]
    node _io_outputValid_T = eq(y, UInt<1>(0h0)) @[src/main/scala/gcd/GCD.scala 36:23]
    connect io.outputValid, _io_outputValid_T @[src/main/scala/gcd/GCD.scala 36:18]
"#;
        let lexer = FIRRTLLexer::new(source);
        let parser = CircuitParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);
    }

    #[test]
    fn circuit_annos() {
        let source =
r#"FIRRTL version 3.3.0
circuit GCD :%[[
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~TestHarness|IntXbar_i1_o1",
    "group":"IntXbar_i1_o1"
  }
]]
  module GCD : @[src/main/scala/gcd/GCD.scala 15:7]
    input clock : Clock @[src/main/scala/gcd/GCD.scala 15:7]
    input reset : UInt<1> @[src/main/scala/gcd/GCD.scala 15:7]
    output io : { flip value1 : UInt<16>, flip value2 : UInt<16>, flip loadingValues : UInt<1>, outputGCD : UInt<16>, outputValid : UInt<1>} @[src/main/scala/gcd/GCD.scala 16:14]

    reg x : UInt, clock @[src/main/scala/gcd/GCD.scala 24:15]
    reg y : UInt, clock @[src/main/scala/gcd/GCD.scala 25:15]
    node _T = gt(x, y) @[src/main/scala/gcd/GCD.scala 27:10]
    when _T : @[src/main/scala/gcd/GCD.scala 27:15]
      node _x_T = sub(x, y) @[src/main/scala/gcd/GCD.scala 27:24]
      node _x_T_1 = tail(_x_T, 1) @[src/main/scala/gcd/GCD.scala 27:24]
      connect x, _x_T_1 @[src/main/scala/gcd/GCD.scala 27:19]
    else :
      node _y_T = sub(y, x) @[src/main/scala/gcd/GCD.scala 28:25]
      node _y_T_1 = tail(_y_T, 1) @[src/main/scala/gcd/GCD.scala 28:25]
      connect y, _y_T_1 @[src/main/scala/gcd/GCD.scala 28:20]
    when io.loadingValues : @[src/main/scala/gcd/GCD.scala 30:26]
      connect x, io.value1 @[src/main/scala/gcd/GCD.scala 31:7]
      connect y, io.value2 @[src/main/scala/gcd/GCD.scala 32:7]
    connect io.outputGCD, x @[src/main/scala/gcd/GCD.scala 35:16]
    node _io_outputValid_T = eq(y, UInt<1>(0h0)) @[src/main/scala/gcd/GCD.scala 36:23]
    connect io.outputValid, _io_outputValid_T @[src/main/scala/gcd/GCD.scala 36:18]
"#;
        let lexer = FIRRTLLexer::new(source);
        let parser = CircuitParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);
    }

    #[test]
    fn nested_whens() {
        let source =
r#"FIRRTL version 3.3.0
circuit NestedWhen :
  module NestedWhen : @[src/main/scala/gcd/NestedWhen.scala 8:7]
    input clock : Clock @[src/main/scala/gcd/NestedWhen.scala 8:7]
    input reset : UInt<1> @[src/main/scala/gcd/NestedWhen.scala 8:7]
    output io : { flip a : UInt<2>, flip b : UInt<2>, flip c : UInt<2>, flip sel : UInt<2>, output : UInt<2>} @[src/main/scala/gcd/NestedWhen.scala 9:14]

    node _T = eq(io.output, UInt<1>(0h0)) @[src/main/scala/gcd/NestedWhen.scala 17:19]
    when _T : @[src/main/scala/gcd/NestedWhen.scala 17:28]
      connect io.output, io.a @[src/main/scala/gcd/NestedWhen.scala 18:15]
    else :
      node _T_1 = eq(io.output, UInt<1>(0h1)) @[src/main/scala/gcd/NestedWhen.scala 19:26]
      when _T_1 : @[src/main/scala/gcd/NestedWhen.scala 19:35]
        connect io.output, io.b @[src/main/scala/gcd/NestedWhen.scala 20:15]
      else :
        connect io.output, io.c @[src/main/scala/gcd/NestedWhen.scala 22:15]

"#;
        let lexer = FIRRTLLexer::new(source);
        let parser = CircuitParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);
    }

    #[test]
    fn one_read_one_write_sram() {
        let source =
r#"FIRRTL version 3.3.0
circuit OneReadOneWritePortSRAM :
  module OneReadOneWritePortSRAM : @[src/main/scala/gcd/SRAM.scala 10:7]
    input clock : Clock @[src/main/scala/gcd/SRAM.scala 10:7]
    input reset : UInt<1> @[src/main/scala/gcd/SRAM.scala 10:7]
    output io : { flip ren : UInt<1>, flip raddr : UInt<3>, rdata : UInt<2>[4], flip wen : UInt<1>, flip waddr : UInt<3>, flip wdata : UInt<2>[4], flip wmask : UInt<1>[4]} @[src/main/scala/gcd/SRAM.scala 11:14]

    smem mem : UInt<2>[4] [8] @[src/main/scala/gcd/SRAM.scala 22:24]
    when io.wen : @[src/main/scala/gcd/SRAM.scala 23:17]
      write mport MPORT = mem[io.waddr], clock @[src/main/scala/gcd/SRAM.scala 24:14]
      when io.wmask[0] : @[src/main/scala/gcd/SRAM.scala 24:14]
        connect MPORT[0], io.wdata[0] @[src/main/scala/gcd/SRAM.scala 24:14]
      when io.wmask[1] : @[src/main/scala/gcd/SRAM.scala 24:14]
        connect MPORT[1], io.wdata[1] @[src/main/scala/gcd/SRAM.scala 24:14]
      when io.wmask[2] : @[src/main/scala/gcd/SRAM.scala 24:14]
        connect MPORT[2], io.wdata[2] @[src/main/scala/gcd/SRAM.scala 24:14]
      when io.wmask[3] : @[src/main/scala/gcd/SRAM.scala 24:14]
        connect MPORT[3], io.wdata[3] @[src/main/scala/gcd/SRAM.scala 24:14]
    wire _WIRE : UInt<3> @[src/main/scala/gcd/SRAM.scala 26:23]
    invalidate _WIRE @[src/main/scala/gcd/SRAM.scala 26:23]
    when io.ren : @[src/main/scala/gcd/SRAM.scala 26:23]
      connect _WIRE, io.raddr @[src/main/scala/gcd/SRAM.scala 26:23]
      read mport MPORT_1 = mem[_WIRE], clock @[src/main/scala/gcd/SRAM.scala 26:23]
    connect io.rdata, MPORT_1 @[src/main/scala/gcd/SRAM.scala 26:12]
    "#;

        let lexer = FIRRTLLexer::new(source);
        let parser = CircuitParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);
    }

    #[test]
    fn as_clock_stmt() {
        let source = "node _childClock_T = asClock(UInt<1>(0h0)) @[generators/diplomacy/diplomacy/src/diplomacy/lazymodule/LazyModuleImp.scala 160:25]";
        let lexer = FIRRTLLexer::new(source);
        let parser = StmtParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);
    }

    #[test]
    fn printf_stmt() {
        let source = r#"printf(clock, UInt<1>(0h1), "Assertion failed: 'A' channel has invalid opcode (connected at generators/rocket-chip/src/main/scala/subsystem/SystemBus.scala:48:55)\n    at Monitor.scala:45 assert(cond, message)\n") : printf @[generators/rocket-chip/src/main/scala/tilelink/Monitor.scala 45:11]"#;
        let lexer = FIRRTLLexer::new(source);
        let parser = StmtParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);
    }

    #[test]
    fn empty_type() {
        let source = "output auto : { } @[generators/diplomacy/diplomacy/src/diplomacy/lazymodule/LazyModuleImp.scala 107:25]";
        let lexer = FIRRTLLexer::new(source);
        let parser = PortParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);
    }

    #[test]
    fn extmodule() {
        let source =
r#"extmodule plusarg_reader : @[generators/rocket-chip/src/main/scala/util/PlusArg.scala 45:7]
     output out : UInt<32>
     defname = plusarg_reader
     parameter DEFAULT = 0
     parameter FORMAT = "tilelink_timeout=%d"
     parameter WIDTH = 32"#;

        let lexer = FIRRTLLexer::new(source);
        let parser = CircuitModuleParser::new();
        let ast = parser.parse(lexer).unwrap();
        println!("{:?}", ast);
    }

    #[test] fn primop_name_overlaps_with_variable() {
        let source =
r#"
node pad = or(bootAddrReg, UInt<64>(0h0)) @[generators/rocket-chip/src/main/scala/regmapper/RegField.scala 150:19]
node _oldBytes_T = bits(pad, 7, 0) @[generators/rocket-chip/src/main/scala/regmapper/RegField.scala 151:57]
node _oldBytes_T_1 = bits(pad, 15, 8) @[generators/rocket-chip/src/main/scala/regmapper/RegField.scala 151:57]
node _oldBytes_T_2 = bits(pad, 23, 16) @[generators/rocket-chip/src/main/scala/regmapper/RegField.scala 151:57]
node _oldBytes_T_3 = bits(pad, 31, 24) @[generators/rocket-chip/src/main/scala/regmapper/RegField.scala 151:57]
node _oldBytes_T_4 = bits(pad, 39, 32) @[generators/rocket-chip/src/main/scala/regmapper/RegField.scala 151:57]
node _oldBytes_T_5 = bits(pad, 47, 40) @[generators/rocket-chip/src/main/scala/regmapper/RegField.scala 151:57]
node _oldBytes_T_6 = bits(pad, 55, 48) @[generators/rocket-chip/src/main/scala/regmapper/RegField.scala 151:57]
node _oldBytes_T_7 = bits(pad, 63, 56) @[generators/rocket-chip/src/main/scala/regmapper/RegField.scala 151:57]
"#;
        let lexer = FIRRTLLexer::new(&source);
        let parser = StmtsParser::new();
        let ast = parser.parse(lexer).expect("FAILED");
        println!("{:?}", ast);
    }

    #[test] fn empty_skip() {
        let source =
r#"
when do_deq : @[src/main/scala/chisel3/util/Decoupled.scala 273:16]
  skip
"#;
        let lexer = FIRRTLLexer::new(&source);
        let parser = StmtsParser::new();
        let ast = parser.parse(lexer).expect("FAILED");
        println!("{:?}", ast);
    }

    #[test]
    fn pad_stmts() {
        let source =
r#"node pad = or(bootAddrReg, UInt<64>(0h0)) @[generators/rocket-chip/src/main/scala/regmapper/RegField.scala 150:19]"#;

        let lexer = FIRRTLLexer::new(&source);
        let parser = StmtsParser::new();
        let ast = parser.parse(lexer).expect("FAILED");
        println!("{:?}", ast);
    }

    #[test]
    fn rocketconfig() -> Result<(), std::io::Error> {
        let source = std::fs::read_to_string("./test-inputs/chipyard.harness.TestHarness.RocketConfig.fir")?;
        let lexer = FIRRTLLexer::new(&source);
        let parser = CircuitParser::new();
        let ast = parser.parse(lexer).expect("FAILED");
    // println!("{:?}", ast);
        Ok(())
    }
}
