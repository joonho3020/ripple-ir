use crate::passes::ast::print::Printer;
use rusty_firrtl::*;

/// Can be used to stringify a FIRRTL3 (legacy FIRRTL) AST:
/// ```rust
/// let mut printer = ChirrtlPrinter::new();
/// let circuit = printer.print_circuit(&ast);
/// ```
pub struct FIRRTL3Printer {
    _indent: u32
}

impl FIRRTL3Printer {
    pub fn new() -> Self {
        Self {
            _indent: 0
        }
    }
}

impl Printer for FIRRTL3Printer {
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
        format!("{}", stmt.to_firrtl_str())
    }
}

trait FIRRTL3Str {
    fn to_firrtl_str(&self) -> String;
}

impl FIRRTL3Str for Int {
    fn to_firrtl_str(&self) -> String {
        format!("\"h{}\"", self.0.to_str_radix(16).to_lowercase())
    }
}

impl FIRRTL3Str for Expr {
    fn to_firrtl_str(&self) -> String {
        match self {
            Expr::UIntNoInit(w) => format!("UInt<{}>()", w),
            Expr::UIntInit(w, init) => format!("UInt<{}>({})", w, init.to_firrtl_str()),
            Expr::SIntNoInit(w) => format!("SInt<{}>()", w),
            Expr::SIntInit(w, init) => format!("SInt<{}>({})", w, init.to_firrtl_str()),
            Expr::Reference(r) => format!("{}", r),
            Expr::Mux(cond, te, fe) => format!("mux({}, {}, {})", cond.to_firrtl_str(), te.to_firrtl_str(), fe.to_firrtl_str()),
            Expr::ValidIf(cond, te) => format!("validif({}, {})", cond.to_firrtl_str(), te.to_firrtl_str()),
            Expr::PrimOp2Expr(op, a, b) => format!("{}({}, {})", op, a.to_firrtl_str(), b.to_firrtl_str()),
            Expr::PrimOp1Expr(op, a) => format!("{}({})", op, a.to_firrtl_str()),
            Expr::PrimOp1Expr1Int(op, a, b) => format!("{}({}, {})", op, a.to_firrtl_str(), b),
            Expr::PrimOp1Expr2Int(op, a, b, c) => format!("{}({}, {}, {})", op, a.to_firrtl_str(), b, c),
        }
    }
}

impl FIRRTL3Str for Stmt {
    fn to_firrtl_str(&self) -> String {
        match self {
            Stmt::Skip(info) => format!("skip {}", info),
            Stmt::Wire(name, tpe, info) => format!("wire {} : {} {}", name, tpe, info),
            Stmt::Reg(name, tpe, clk, info) => {
                if info == &Info::default() {
                    format!("reg {} : {}, {} with :\n  reset => (UInt<1>(\"h0\"), {})", name, tpe, clk.to_firrtl_str(), name)
                } else {
                    format!("reg {} : {}, {} with :\n  reset => (UInt<1>(\"h0\"), {}) {}", name, tpe, clk.to_firrtl_str(), name, info)
                }
            }
            Stmt::RegReset(name, tpe, clk, rst, init, info) => {
                if info == &Info::default() {
                    format!("reg {} : {}, {} with :\n  reset => ({}, {})", name, tpe, clk.to_firrtl_str(), rst.to_firrtl_str(), init.to_firrtl_str())
                } else {
                    format!("reg {} : {}, {} with :\n  reset => ({}, {}) {}", name, tpe, clk.to_firrtl_str(), rst.to_firrtl_str(), init.to_firrtl_str(), info)
                }
            }
            Stmt::ChirrtlMemory(cm) => format!("{}", cm),
            Stmt::ChirrtlMemoryPort(cmp) => format!("{}", cmp),
            Stmt::Inst(inst, module, info) => {
                if info == &Info::default() {
                    format!("inst {} of {}", inst, module)
                } else {
                    format!("inst {} of {} {}", inst, module, info)
                }
            }
            Stmt::Node(name, expr, info) => {
                if info == &Info::default() {
                    format!("node {} = {}", name, expr.to_firrtl_str())
                } else {
                    format!("node {} = {} {}", name, expr.to_firrtl_str(), info)
                }
            }
            Stmt::Connect(lhs, rhs, info) => {
                if info == &Info::default() {
                    format!("{} <= {}", lhs.to_firrtl_str(), rhs.to_firrtl_str())
                } else {
                    format!("{} <= {} {}", lhs.to_firrtl_str(), rhs.to_firrtl_str(), info)
                }
            }
            Stmt::Invalidate(reference, info) => format!("{} is invalid {}", reference.to_firrtl_str(), info),
            Stmt::Printf(name_opt, clk, clk_val, msg, fields_opt, info) => {
                if let Some(fields) = fields_opt {
                    if let Some(name) = name_opt {
                        format!("printf({}, {}, {}, {}) : {} {}", clk.to_firrtl_str(), clk_val.to_firrtl_str(), msg, fmt_exprs_as_msg(fields), name, info)
                    } else {
                        format!("printf({}, {}, {}, {}) : {}", clk.to_firrtl_str(), clk_val.to_firrtl_str(), msg, fmt_exprs_as_msg(fields), info)
                    }
                } else {
                    if let Some(name) = name_opt {
                        format!("printf({}, {}, {}) : {} {}", clk.to_firrtl_str(), clk_val.to_firrtl_str(), msg, name, info)
                    } else {
                        format!("printf({}, {}, {}) : {}", clk.to_firrtl_str(), clk_val.to_firrtl_str(), msg, info)
                    }
                }
            }
            Stmt::Assert(name_opt, clk, cond, cond_val, msg, info) => {
                if let Some(name) = name_opt {
                    format!("assert({}, {}, {}, {}) : {} {}", clk.to_firrtl_str(), cond.to_firrtl_str(), cond_val.to_firrtl_str(), msg, name, info)
                } else {
                    format!("assert({}, {}, {}, {}) : {}", clk.to_firrtl_str(), cond.to_firrtl_str(), cond_val.to_firrtl_str(), msg, info)
                }
            }
            Stmt::When(_cond, _info, _when_stmts, _else_stmts_opt) => {
                unimplemented!()
            }
            Stmt::Stop(name, clk, cond, x, info) => {
                format!("stop({}, {}, {}) : {} {}", clk, cond, x, name, info)
            }
            Stmt::Memory(name, tpe, depth, rlat, wlat, ports, _ruw, info) => {
                let mut ret = String::new();
                ret.push_str(&format!("mem {} : {}\n", name, info));
                ret.push_str(&format!("  data-type => {}\n", tpe));
                ret.push_str(&format!("  depth => {}\n", depth));
                ret.push_str(&format!("  read-latency => {}\n", rlat));
                ret.push_str(&format!("  write-latency => {}\n", wlat));
                for port in ports {
                    match port.as_ref() {
                        MemoryPort::Write(name) => {
                            ret.push_str(&format!("  writer => {}\n", name));
                        }
                        MemoryPort::Read(name) => {
                            ret.push_str(&format!("  reader => {}\n", name));
                        }
                        MemoryPort::ReadWrite(name) => {
                            ret.push_str(&format!("  readwriter => {}\n", name));
                        }
                    }
                }
                // NOTE: ruw is always undefined in all cases that we care about
                ret.push_str(&format!("  read-under-write => undefined"));
                ret
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::common::RippleIRErr;
    use test_case::test_case;
    use firrtl3_parser::parse_circuit;

    #[test_case("Adder" ; "Adder")]
    #[test_case("GCD" ; "GCD")]
    #[test_case("GCDDelta" ; "GCDDelta")]
    #[test_case("Fir" ; "Fir")]
    #[test_case("BitSel1" ; "BitSel1")]
    #[test_case("BitSel2" ; "BitSel2")]
    #[test_case("LCS1" ; "LCS1")]
    #[test_case("LCS2" ; "LCS2")]
    #[test_case("LCS3" ; "LCS3")]
    #[test_case("LCS4" ; "LCS4")]
    #[test_case("LCS5" ; "LCS5")]
    #[test_case("LCS6" ; "LCS6")]
    #[test_case("LCS7" ; "LCS7")]
    #[test_case("LCS8" ; "LCS8")]
    #[test_case("CombHierarchy" ; "CombHierarchy")]
    #[test_case("DecoupledMux" ; "DecoupledMux")]
    #[test_case("DynamicIndexing" ; "DynamicIndexing")]
    #[test_case("RegFile" ; "RegFile")]
    #[test_case("RegInitWire" ; "RegInitWire")]
    #[test_case("RegVecInit" ; "RegVecInit")]
    #[test_case("Subtracter" ; "Subtracter")]
    #[test_case("Top" ; "Top")]
// #[test_case("MultiWhen" ; "MultiWhen")]
// #[test_case("MyQueue" ; "MyQueue")]
// #[test_case("NestedWhen" ; "NestedWhen")]
// #[test_case("SinglePortSRAM" ; "SinglePortSRAM")]
// #[test_case("DualReadSingleWritePortSRAM" ; "DualReadSingleWritePortSRAM")]
// #[test_case("OneReadOneReadWritePortSRAM" ; "OneReadOneReadWritePortSRAM")]
// #[test_case("OneReadOneWritePortSRAM" ; "OneReadOneWritePortSRAM")]
// #[test_case("Cache" ; "Cache")]
// #[test_case("PointerChasing" ; "PointerChasing")]
// #[test_case("FireSimRocket" ; "FireSimRocket")]
// #[test_case("FireSimLargeBoom" ; "FireSimLargeBoom")]
    fn run(name: &str) -> Result<(), RippleIRErr> {
        let file_path = format!("./test-inputs-firrtl3/{}.fir", name);
        let circuit_str = std::fs::read_to_string(file_path)?;
        let circuit = parse_circuit(&circuit_str).expect("firrtl parser");

        let mut printer = FIRRTL3Printer::new();
        let reconstructed_circuit_str = printer.print_circuit(&circuit);

        if circuit_str != reconstructed_circuit_str {
            let out_path = format!("./test-outputs/{}.firrtl3.fir", name);
            std::fs::write(&out_path, reconstructed_circuit_str)?;
            Err(RippleIRErr::MiscError("Reconstructed FIRRTL3 differs".to_string()))
        } else {
            Ok(())
        }
    }
}
