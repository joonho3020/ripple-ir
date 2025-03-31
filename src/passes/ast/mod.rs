pub mod check_ast_assumption;
pub mod print;
pub mod infer_readwrite;

#[cfg(test)]
mod test {
    use chirrtl_parser::parse_circuit;
    use crate::common::RippleIRErr;
    use crate::passes::ast::print::Printer;
    use crate::passes::ast::check_ast_assumption::check_ast_assumption;

    #[test]
    fn check_printer() {
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
        let ast = parse_circuit(&source).expect("Parse failed");
        let mut printer = Printer::new();
        let circuit = printer.print_circuit(&ast);
        let expect =
r#"FIRRTL Version 3.3.0
circuit NestedWhen :%[[
Annotations(
    Null,
)
]]
  module NestedWhen : @[src/main/scala/gcd/NestedWhen.scala 8:7]
    input clock : Clock @[src/main/scala/gcd/NestedWhen.scala 8:7]
    input reset : UInt<1> @[src/main/scala/gcd/NestedWhen.scala 8:7]
    output io : { flip a: UInt<2>, flip b: UInt<2>, flip c: UInt<2>, flip sel: UInt<2>, output: UInt<2>,  } @[src/main/scala/gcd/NestedWhen.scala 9:14]

    node _T = Eq(io.output, UInt<1>(Int(0))) @[src/main/scala/gcd/NestedWhen.scala 17:19]
    when _T : @[src/main/scala/gcd/NestedWhen.scala 17:28]
      connect io.output, io.a @[src/main/scala/gcd/NestedWhen.scala 18:15]
    else :
      node _T_1 = Eq(io.output, UInt<1>(Int(1))) @[src/main/scala/gcd/NestedWhen.scala 19:26]
      when _T_1 : @[src/main/scala/gcd/NestedWhen.scala 19:35]
        connect io.output, io.b @[src/main/scala/gcd/NestedWhen.scala 20:15]
      else :
        connect io.output, io.c @[src/main/scala/gcd/NestedWhen.scala 22:15]

"#;
      assert_eq!(circuit, expect);
    }

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
