


#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Width(pub u32);

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Keyword {
    Inst,
    Printf,
    Assert,
    SMem,
    CMem,
    Of,
    Reg,
    Input,
    Output,
    Invalidate,
    Mux,
    Stop,
    Depth,
    Write,
    Read,
    Version,
    Probe,
    Module,
    Const,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Identifier {
    ID(u32),
    Name(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Reference {
    Ref(Identifier),
    RefDot(Box<Reference>, Identifier),
    RefIdxInt(Box<Reference>, Identifier)
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
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


#[derive(Debug, Clone, PartialEq)]
pub enum PrimOp1Expr1Int {
    Pad,
    Shl,
    Shr,
    Head,
    Tail,
    BitSel,
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    IntType(Width),
    Reference(Reference),
    PrimOp2Expr(PrimOp2Expr, Box<Expr>, Box<Expr>),
    PrimOp1Expr(PrimOp1Expr, Box<Expr>),
    PrimOp1Expr1Int(PrimOp1Expr1Int, Box<Expr>, u32),
    PrimOp1Expr2Int(PrimOp1Expr2Int, Box<Expr>, u32, u32),
}

// #[derive(Debug, Clone)]
// pub struct Circuit {
// pub identifier: String,
// pub modules: Vec<Module>
// }

// #[derive(Debug, Clone)]
// pub struct Module {
// pub identifier: String,
// pub ports: Vec<Port>,
// pub stmts: Vec<Statements>
// }

// #[derive(Debug, Clone)]
// pub struct Port {
// }
