use std::fmt::Display;
use crate::parser::Int;

#[derive(Debug, Default, Clone, PartialEq, Hash)]
pub struct Info(pub String);

impl Display for Info {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "@[{}]", self.0)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub struct Width(pub u32);

impl Display for Width {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Identifier {
    ID(Int),
    Name(String),
}

impl Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ID(x)   => write!(f, "{:?}", x),
            Self::Name(x) => write!(f, "{}", x)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct Float {
    pub integer: u32,
    pub decimal: u32,
}

impl Float {
    pub fn new(integer: u32, decimal: u32) -> Self {
        Self { integer, decimal }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Reference {
    Ref(Identifier),
    RefDot(Box<Reference>, Identifier),
    RefIdxInt(Box<Reference>, Int),
    RefIdxExpr(Box<Reference>, Box<Expr>)
}

impl Display for Reference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ref(name)    => write!(f, "{}", name),
            Self::RefDot(r, name) => write!(f, "{}.{}", r, name),
            Self::RefIdxInt(r, int) => write!(f, "{}[{:?}]", r, int),
            Self::RefIdxExpr(r, expr) => write!(f, "{}[{}]", r, expr),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum PrimOp2Expr {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Lt,
    Leq,
    Gt,
    Geq,
    Eq,
    Neq,
    Dshl,
    Dshr,
    And,
    Or,
    Xor,
    Cat,
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum PrimOp1Expr {
    AsUInt,
    AsSInt,
    AsClock,
    AsAsyncReset,
    Cvt,
    Neg,
    Not,
    Andr,
    Orr,
    Xorr,
}


#[derive(Debug, Clone, PartialEq, Hash)]
pub enum PrimOp1Expr1Int {
    Pad,
    Shl,
    Shr,
    Head,
    Tail,
    BitSel,
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum PrimOp1Expr2Int {
    BitSelRange,
}

impl From<String> for PrimOp2Expr {
    fn from(value: String) -> Self {
        match value.as_str() {
            "add" => { Self::Add },
            "sub" => { Self::Sub },
            "mul" => { Self::Mul },
            "div" => { Self::Div },
            "rem" => { Self::Rem },
            "lt"  => { Self::Lt },
            "leq"  => { Self::Leq },
            "gt"  => { Self::Gt },
            "geq"  => { Self::Geq },
            "eq"  => { Self::Eq },
            "neq"  => { Self::Neq },
            "dshl"  => { Self::Dshl },
            "dshr"  => { Self::Dshr },
            "and"  => { Self::And },
            "or"  => { Self::Or },
            "xor"  => { Self::Xor },
            "cat"  => { Self::Cat },
            _ => {
                panic!("Unrecognized operator {}", value);
            }
        }
    }
}

impl From<String> for PrimOp1Expr {
    fn from(value: String) -> Self {
        match value.as_str() {
            "asUInt"  => { Self::AsUInt },
            "asSInt"  => { Self::AsSInt },
            "asClock"  => { Self::AsClock },
            "asAsyncReset"  => { Self::AsAsyncReset },
            "cvt"  => { Self::Cvt },
            "neg"  => { Self::Neg },
            "not"  => { Self::Not },
            "andr"  => { Self::Andr },
            "orr"  => { Self::Orr },
            "xorr"  => { Self::Xorr },
            _ => {
                panic!("Unrecognized operator {}", value);
            }
        }
    }
}

impl From<String> for PrimOp1Expr1Int {
    fn from(value: String) -> Self {
        match value.as_str() {
            "pad"  => { Self::Pad },
            "shl"  => { Self::Shl },
            "shr"  => { Self::Shr },
            "head"  => { Self::Head },
            "tail"  => { Self::Tail },
            _ => { Self::BitSel }
        }
    }
}

impl From<String> for PrimOp1Expr2Int {
    fn from(value: String) -> Self {
        assert!(value.contains("bit"));
        Self::BitSelRange
    }
}

pub type Exprs = Vec<Box<Expr>>;

fn fmt_exprs(exprs: &Exprs) -> String {
    let mut ret = "".to_string();
    let len = exprs.len();
    for (id, e) in exprs.iter().enumerate() {
        ret.push_str(&format!("{}", e));
        if id != len - 1 {
            ret.push_str(", ");
        }
    }
    return ret;
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Expr {
    UIntNoInit(Width),
    UIntInit(Width, Int),
    SIntNoInit(Width),
    SIntInit(Width, Int),
    Reference(Reference),
    Mux(Box<Expr>, Box<Expr>, Box<Expr>),
    ValidIf(Box<Expr>, Box<Expr>),
    PrimOp2Expr(PrimOp2Expr, Box<Expr>, Box<Expr>),
    PrimOp1Expr(PrimOp1Expr, Box<Expr>),
    PrimOp1Expr1Int(PrimOp1Expr1Int, Box<Expr>, Int),
    PrimOp1Expr2Int(PrimOp1Expr2Int, Box<Expr>, Int, Int),
}

impl Expr {
    pub fn parse_radixint(s: &str) -> Result<Int, String> {
        if let Some(num) = s.strip_prefix("0b") {
            Int::from_str_radix(num, 2).map_err(|e| e.to_string())
        } else if let Some(num) = s.strip_prefix("0o") {
            Int::from_str_radix(num, 8).map_err(|e| e.to_string())
        } else if let Some(num) = s.strip_prefix("0d") {
            Int::from_str_radix(num, 10).map_err(|e| e.to_string())
        } else if let Some(num) = s.strip_prefix("0h") {
            Int::from_str_radix(num, 16).map_err(|e| e.to_string())
        } else {
            Err(format!("Invalid number format: {}", s))
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::UIntNoInit(w) => write!(f, "UInt<{}>()", w),
            Expr::UIntInit(w, init) => write!(f, "UInt<{}>({:?})", w, init),
            Expr::SIntNoInit(w) => write!(f, "SInt<{}>()", w),
            Expr::SIntInit(w, init) => write!(f, "SInt<{}>({:?})", w, init),
            Expr::Reference(r) => write!(f, "{}", r),
            Expr::Mux(cond, te, fe) => write!(f, "mux({}, {}, {})", cond, te, fe),
            Expr::ValidIf(cond, te) => write!(f, "validif({}, {})", cond, te),
            Expr::PrimOp2Expr(op, a, b) => write!(f, "{:?}({}, {})", op, a, b),
            Expr::PrimOp1Expr(op, a) => write!(f, "{:?}({})", op, a),
            Expr::PrimOp1Expr1Int(op, a, b) => write!(f, "{:?}({}, {:?})", op, a, b),
            Expr::PrimOp1Expr2Int(op, a, b, c) => write!(f, "{:?}({}, {:?}, {:?})", op, a, b, c),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum TypeGround {
    Clock,
    Reset,
    AsyncReset,
    UInt(Option<Width>),
    SInt(Option<Width>),
// ProbeType
// AnalType,
// FixedType
}

impl Display for TypeGround {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Clock => { write!(f, "Clock") }
            Self::Reset => { write!(f, "Reset") }
            Self::AsyncReset => { write!(f, "AsyncReset") }
            Self::UInt(w_opt) => {
                if let Some(w) = w_opt {
                    write!(f, "UInt<{}>", w)
                } else {
                    write!(f, "UInt")
                }
            }
            Self::SInt(w_opt) => {
                if let Some(w) = w_opt {
                    write!(f, "SInt<{}>", w)
                } else {
                    write!(f, "SInt")
                }
            }
        }
    }
}

pub type Fields = Vec<Box<Field>>;

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Field {
    Straight(Identifier, Box<Type>),
    Flipped(Identifier, Box<Type>),
}

impl Display for Field {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Straight(id, tpe) => {
                write!(f, "{}: {}", id, tpe)
            }
            Self::Flipped(id, tpe) => {
                write!(f, "flip {}: {}", id, tpe)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum TypeAggregate {
    Fields(Box<Fields>),
    Array(Box<Type>, Int),
}

impl Display for TypeAggregate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fields(fields) => {
                write!(f, "{{ ")?;
                for field in fields.iter() {
                    write!(f, "{}, ", field)?;
                }
                write!(f, " }}")
            }
            Self::Array(tpe, idx) => {
                write!(f, "{}[{:?}]", tpe, idx)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Type {
    TypeGround(TypeGround),
    ConstTypeGround(TypeGround),
    TypeAggregate(Box<TypeAggregate>),
    ConstTypeAggregate(Box<TypeAggregate>),
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeGround(tg) => write!(f, "{}", tg),
            Self::ConstTypeGround(tg) => write!(f, "{}", tg),
            Self::TypeAggregate(ta) => write!(f, "{}", ta),
            Self::ConstTypeAggregate(ta) => write!(f, "{}", ta),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Hash)]
pub enum ChirrtlMemoryReadUnderWrite {
    #[default]
    Undefined,
    Old,
    New
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum ChirrtlMemory {
    SMem(Identifier, Type, Option<ChirrtlMemoryReadUnderWrite>, Info),
    CMem(Identifier, Type, Info),
}

impl Display for ChirrtlMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SMem(name, tpe, ruw_opt, info) => {
                if let Some(ruw) = ruw_opt {
                    write!(f, "smem {} : {}, {:?} {}", name, tpe, ruw, info)
                } else {
                    write!(f, "smem {} : {} {}", name, tpe, info)
                }
            }
            Self::CMem(name, tpe, info) => {
                    write!(f, "cmem {} : {} {}", name, tpe, info)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum ChirrtlMemoryPort {
    Write(Identifier, Identifier, Expr, Reference, Info),
    Read (Identifier, Identifier, Expr, Reference, Info),
    Infer(Identifier, Identifier, Expr, Reference, Info),
}

impl Display for ChirrtlMemoryPort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChirrtlMemoryPort::Write(port_name, mem_name, addr, clk, info) => {
                write!(f, "write mport {} = {}[{}], {} {}", port_name, mem_name, addr, clk, info)
            }
            ChirrtlMemoryPort::Read(port_name, mem_name, addr, clk, info) => {
                write!(f, "read mport {} = {}[{}], {} {}", port_name, mem_name, addr, clk, info)
            }
            ChirrtlMemoryPort::Infer(port_name, mem_name, addr, clk, info) => {
                write!(f, "infer mport {} = {}[{}], {} {}", port_name, mem_name, addr, clk, info)
            }
        }
    }
}

pub type Stmts = Vec<Box<Stmt>>;

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Stmt {
    Skip(Info),
    Wire(Identifier, Type, Info),
    Reg(Identifier,  Type, Expr, Info),
    RegReset(Identifier, Type, Expr, Expr, Expr, Info),
// Memory()
    ChirrtlMemory(ChirrtlMemory),
    ChirrtlMemoryPort(ChirrtlMemoryPort),
    Inst(Identifier, Identifier, Info),
    Node(Identifier, Expr, Info),
    Connect(Expr, Expr, Info),
// Connect(Reference, Read, Expr, Info),
// Reference <- ???
    Invalidate(Expr, Info),
// Define(Define, Reference, Probe, Info),
// Define(Define, Reference, Expr, Probe, Info),
// Attach(References)
    When(Expr, Info, Stmts, Option<Stmts>),
// Stop(Expr, Expr, u64, Info),
// Stop(Expr, Expr, u64, Info),
    Printf(Expr, Expr, String, Option<Exprs>, Info),
    Assert(Expr, Expr, Expr, String, Info),
}

impl Stmt {
    pub fn traverse(&self) {
        match self {
            Self::When(e, i, tstmts, fstmts_opt) => {
                println!("When, {:?}, {:?}", e, i);
                for tstmt in tstmts.iter() {
                    println!("{:?}", tstmt);
                }
                match fstmts_opt {
                    Some(fstmts) => {
                        println!("ELSE");
                        for fstmt in fstmts.iter() {
                            println!("{:?}", fstmt);
                        }
                    }
                    None => {
                    }
                }
            }
            _ => {
                println!("{:?}", self);
            }
        }
    }
}

impl Display for Stmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stmt::Skip(info) => write!(f, "skip {}", info),
            Stmt::Wire(name, tpe, info) => write!(f, "wire {} : {} {}", name, tpe, info),
            Stmt::Reg(name, tpe, clk, info) => write!(f, "reg {} : {}, {} {}", name, tpe, clk, info),
            Stmt::RegReset(name, tpe, clk, rst, init, info) => write!(f, "reg {} : {}, {}, {}, {} {}", name, tpe, clk, rst, init, info),
            Stmt::ChirrtlMemory(cm) => write!(f, "{}", cm),
            Stmt::ChirrtlMemoryPort(cmp) => write!(f, "{}", cmp),
            Stmt::Inst(inst, module, info) => write!(f, "{} of {} {}", inst, module, info),
            Stmt::Node(name, expr, info) => write!(f, "node {} = {} {}", name, expr, info),
            Stmt::Connect(lhs, rhs, info) => write!(f, "connect {}, {} {}", lhs, rhs, info),
            Stmt::Invalidate(reference, info) => write!(f, "invalidate {} {}", reference, info),
            Stmt::Printf(clk, clk_val, msg, fields_opt, info) => {
                if let Some(fields) = fields_opt {
                    write!(f, "printf({}, {}, {}, {}) : {}", clk, clk_val, msg, fmt_exprs(fields), info)
                } else {
                    write!(f, "printf({}, {}, {}) : {}", clk, clk_val, msg, info)
                }
            }
            Stmt::Assert(clk, cond, cond_val, msg, info) => {
                write!(f, "assert({}, {}, {}, {}) : {}", clk, cond, cond_val, msg, info)
            }
            Stmt::When(cond, info, when_stmts, else_stmts_opt) => {
                // NOTE: Cannot track indent inside the Display trait, so this will cause weirdly
                // indented stuff
                writeln!(f, "when {} : {}", cond, info)?;
                for stmt in when_stmts.iter() {
                    writeln!(f, "{}{}", " ".repeat(2), stmt)?;
                }
                if let Some(else_stmts) = else_stmts_opt {
                    writeln!(f, "else :")?;
                    for stmt in else_stmts.iter() {
                        writeln!(f, "{}{}", " ".repeat(2), stmt)?;
                    }
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Port {
    Input(Identifier, Type, Info),
    Output(Identifier, Type, Info),
}

impl Display for Port {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input(id, tpe, info) => {
                write!(f, "input {} : {} {}", id, tpe, info)
            }
            Self::Output(id, tpe, info) => {
                write!(f, "output {} : {} {}", id, tpe, info)
            }
        }
    }
}

pub type Ports = Vec<Box<Port>>;

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct Module  {
    pub name: Identifier,
    pub ports: Ports,
    pub stmts: Stmts,
    pub info: Info,
}

impl Module {
    pub fn new(name: Identifier, ports: Ports, stmts: Stmts, info: Info) -> Self {
        Self { name, ports, stmts, info, }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct DefName(Identifier);

impl From<Identifier> for DefName {
    fn from(value: Identifier) -> Self {
        Self(value)
    }
}

pub type Parameters = Vec<Box<Parameter>>;

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Parameter {
    IntParam(Identifier, Int),
    FloatParam(Identifier, Float),
    StringParam(Identifier, String),
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct ExtModule {
    pub name: Identifier,
    pub ports: Ports,
    pub defname: DefName,
    pub params: Parameters,
    pub info: Info,
}

impl ExtModule {
    pub fn new(name: Identifier, ports: Ports, defname: DefName, params: Parameters, info: Info) -> Self {
        Self { name, ports, defname, params, info }
    }
}

#[allow(dead_code)]
pub struct IntModule {
    pub name: Identifier,
    pub ports: Ports,
    pub params: Parameters,
    pub info: Info,
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum CircuitModule {
    Module(Module),
    ExtModule(ExtModule),
// IntModule(IntModule),
}

pub type CircuitModules = Vec<Box<CircuitModule>>;

#[derive(Debug, Default, Clone, PartialEq, Hash)]
pub struct Annotations(pub serde_json::Value);

impl Annotations {
    pub fn from_str(input: String) -> Self {
        let input = "[".to_owned() + &input + "]";
        Self { 0: serde_json::from_str(&input).unwrap() }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Hash)]
pub struct Version(pub u32, pub u32, pub u32);

impl Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Version {}.{}.{}", self.0, self.1, self.2)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Circuit {
    pub version: Version,
    pub name: Identifier,
    pub annos: Annotations,
    pub modules: CircuitModules,
}

impl Circuit {
    pub fn new(version: Version, name: Identifier, annos: Annotations, modules: CircuitModules) -> Self {
        Self { version, name, annos, modules }
    }
}
