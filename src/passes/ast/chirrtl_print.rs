use crate::passes::ast::print::Printer;

/// Can be used to stringify a CHIRRTL AST:
/// ```rust
/// let mut printer = ChirrtlPrinter::new();
/// let circuit = printer.print_circuit(&ast);
/// ```
pub struct ChirrtlPrinter {
    _indent: u32,
}

impl ChirrtlPrinter {
    pub fn new() -> Self {
        Self {
            _indent: 0
        }
    }
}

impl Printer for ChirrtlPrinter {
    fn indent(&self) -> u32 {
        self._indent
    }

    fn add_indent(&mut self) {
        self._indent += 1;
    }

    fn add_dedent(&mut self) {
        self._indent -= 1;
    }

    fn stmt_str(&self, stmt: &rusty_firrtl::Stmt) -> String {
        format!("{}", stmt)
    }
}
