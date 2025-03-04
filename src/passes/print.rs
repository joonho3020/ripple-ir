use crate::parser::ast::*;

pub struct Printer {
    indent: u32,
}

impl Printer {
    pub fn new() -> Self {
        Self {
            indent: 0
        }
    }

    fn indent(&mut self) {
        self.indent += 1;
    }

    fn dedent(&mut self) {
        self.indent -= 1;
    }

    pub fn print_circuit(&mut self, circuit: &Circuit) {
        println!("FIRRTL {}", circuit.version);
        println!("circuit {} :%[[", circuit.name);
        println!("{:#?}", circuit.annos);
        println!("]]");

        self.indent();

        for module in circuit.modules.iter() {
            self.print_circuit_module(&module);
        }
    }

    fn print_circuit_module(&mut self, module: &CircuitModule) {
        match module {
            CircuitModule::Module(m) => {
                self.print_module(m);
            }
            CircuitModule::ExtModule(e) => {
            }
        }
    }

    fn print_module(&mut self, module: &Module) {
        self.println_indent(format!("module {} : {}", module.name, module.info));
        self.indent();

        self.print_ports(&module.ports);
        println!("");

        self.print_stmts(&module.stmts);


    }

    fn print_ports(&mut self, ports: &Ports) {
        for port in ports.iter() {
            self.println_indent(format!("{}", port));
        }
    }

    fn print_stmts(&mut self, stmts: &Stmts) {
        for stmt in stmts {
            match stmt.as_ref() {
                Stmt::When(cond, info, when_stmts, else_stmts_opt) => {
                    self.println_indent(format!("when {} : {}", cond, info));
                    self.indent();
                    self.print_stmts(when_stmts);
                    self.dedent();

                    if let Some(else_stmts) = else_stmts_opt {
                        self.println_indent("else :".to_string());
                        self.indent();
                        self.print_stmts(else_stmts);
                        self.dedent();
                    }
                }
                _ => {
                    self.println_indent(format!("{}", stmt));
                }
            }
        }
    }

    fn println_indent(&self, str: String) {
        println!("{}{}", " ".repeat((self.indent * 2) as usize), str);
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::parser::parse_circuit;

    #[test]
    fn gcd() {
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
        let ast = parse_circuit(&source).expect("Parse failed");
        let mut printer = Printer::new();
        printer.print_circuit(&ast);
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
        let ast = parse_circuit(&source).expect("Parse failed");
        let mut printer = Printer::new();
        printer.print_circuit(&ast);
    }
}
