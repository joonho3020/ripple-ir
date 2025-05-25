use rusty_firrtl::*;
use serde_json::Value;

/// Can be used to stringify a FIRRTL AST:
/// ```rust
/// let mut printer = ChirrtlPrinter::new();
/// let circuit = printer.print_circuit(&ast);
/// ```
pub struct ChirrtlPrinter {
    indent: u32,
}

impl ChirrtlPrinter {
    pub fn new() -> Self {
        Self {
            indent: 0
        }
    }

    pub fn print_circuit(&mut self, circuit: &Circuit) -> String {
        let mut ret = "".to_string();
        ret.push_str(&format!("FIRRTL {}\n", circuit.version));

        let annos_is_empty = match &circuit.annos.0 {
            serde_json::Value::Null => true,
            serde_json::Value::Array(arr) => arr.is_empty(),
            serde_json::Value::Object(map) => map.is_empty(),
            _ => false,
        };
        if annos_is_empty {
            ret.push_str(&format!("circuit {} :\n", circuit.name));
        } else {
            ret.push_str(&format!("circuit {} :%[[\n", circuit.name));
            if let Value::Array(outer_array) = &circuit.annos.0 {
                if let Some(inner_value) = outer_array.get(0) {
                    if let Value::Array(inner_array) = inner_value {
                        ret.push_str(&format!("{}\n", serde_json::to_string_pretty(&inner_array.get(0)).unwrap()));
                    }
                }
            }
            ret.push_str(&format!("]]\n"));
        }

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
        self.dedent();
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
        self.dedent();
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
