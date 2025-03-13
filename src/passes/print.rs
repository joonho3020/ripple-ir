use chirrtl_parser::ast::*;

/// Can be used to stringify a FIRRTL AST:
/// ```rust
/// let mut printer = Printer::new();
/// let circuit = printer.print_circuit(&ast);
/// ```
pub struct Printer {
    indent: u32,
}

impl Printer {
    pub fn new() -> Self {
        Self {
            indent: 0
        }
    }

    pub fn print_circuit(&mut self, circuit: &Circuit) -> String {
        let mut ret = "".to_string();
        ret.push_str(&format!("FIRRTL {}\n", circuit.version));
        ret.push_str(&format!("circuit {} :%[[\n", circuit.name));
        ret.push_str(&format!("{:#?}\n", circuit.annos));
        ret.push_str(&format!("]]\n"));

        self.indent();

        for module in circuit.modules.iter() {
            self.print_circuit_module(&module, &mut ret);
        }
        return ret;
    }

    fn indent(&mut self) {
        self.indent += 1;
    }

    fn dedent(&mut self) {
        self.indent -= 1;
    }

    fn print_circuit_module(&mut self, module: &CircuitModule, ret: &mut String) {
        match module {
            CircuitModule::Module(m) => {
                self.print_module(m, ret);
            }
            CircuitModule::ExtModule(e) => {
                self.print_ext_module(e, ret);
            }
        }
        ret.push_str("\n");
    }

    fn print_module(&mut self, module: &Module, ret: &mut String) {
        ret.push_str(&self.println_indent(format!("module {} : {}", module.name, module.info)));
        self.indent();

        self.print_ports(&module.ports, ret);
        ret.push_str("\n");

        self.print_stmts(&module.stmts, ret);
    }

    fn print_ext_module(&mut self, ext: &ExtModule, ret: &mut String) {
        ret.push_str(&self.println_indent(format!("extmodule {} : {}", ext.name, ext.info)));
        self.indent();

        self.print_ports(&ext.ports, ret);
        ret.push_str("\n");

        ret.push_str(&self.println_indent(format!("{}", ext.defname)));
        for param in ext.params.iter() {
            ret.push_str(&self.println_indent(format!("{}", param)));
        }
    }

    fn print_ports(&mut self, ports: &Ports, ret: &mut String) {
        for port in ports.iter() {
            ret.push_str(&self.println_indent(format!("{}", port)));
        }
    }

    /// Recursively print the statements
    fn print_stmts(&mut self, stmts: &Stmts, ret: &mut String) {
        for stmt in stmts {
            match stmt.as_ref() {
                Stmt::When(cond, info, when_stmts, else_stmts_opt) => {
                    ret.push_str(&self.println_indent(format!("when {} : {}", cond, info)));
                    self.indent();
                    self.print_stmts(when_stmts, ret);
                    self.dedent();

                    if let Some(else_stmts) = else_stmts_opt {
                        ret.push_str(&self.println_indent("else :".to_string()));
                        self.indent();
                        self.print_stmts(else_stmts, ret);
                        self.dedent();
                    }
                }
                _ => {
                    ret.push_str(&self.println_indent(format!("{}", stmt)));
                }
            }
        }
    }

    fn println_indent(&self, str: String) -> String {
        format!("{}{}\n", " ".repeat((self.indent * 2) as usize), str)
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use chirrtl_parser::parse_circuit;

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
}
