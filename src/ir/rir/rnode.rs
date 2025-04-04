use chirrtl_parser::ast::*;
use derivative::Derivative;
use std::hash::Hash;
use crate::ir::whentree::*;
use crate::ir::fir::*;
use crate::impl_clean_display;
use crate::define_index_type;
use crate::ir::typetree::tnode::*;

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub enum RippleNodeType {
    #[default]
    Invalid,

    DontCare,

    UIntLiteral(Int),
    SIntLiteral(Int),

    Mux,
    PrimOp2Expr(PrimOp2Expr),
    PrimOp1Expr(PrimOp1Expr),
    PrimOp1Expr1Int(PrimOp1Expr1Int, Int),
    PrimOp1Expr2Int(PrimOp1Expr2Int, Int, Int),

    // Stmt
    Wire,
    Reg,
    RegReset,
    SMem(Option<ChirrtlMemoryReadUnderWrite>),
    CMem,
    WriteMemPort(PrioritizedConds),
    ReadMemPort(PrioritizedConds),
    InferMemPort(PrioritizedConds),
    Inst(Identifier),

    // Port
    Input,
    Output,
    Phi,
}

impl_clean_display!(RippleNodeType);

impl From<&FirNodeType> for RippleNodeType {
    fn from(value: &FirNodeType) -> Self {
        match value {
            FirNodeType::Invalid => Self::Invalid,
            FirNodeType::DontCare => Self::DontCare,
            FirNodeType::UIntLiteral(_, x) => Self::UIntLiteral(x.clone()),
            FirNodeType::SIntLiteral(_, x) => Self::SIntLiteral(x.clone()),
            FirNodeType::Mux => Self::Mux,
            FirNodeType::PrimOp2Expr(op) => Self::PrimOp2Expr(op.clone()),
            FirNodeType::PrimOp1Expr(op) => Self::PrimOp1Expr(op.clone()),
            FirNodeType::PrimOp1Expr1Int(op, a) => Self::PrimOp1Expr1Int(op.clone(), a.clone()),
            FirNodeType::PrimOp1Expr2Int(op, a, b) => Self::PrimOp1Expr2Int(op.clone(), a.clone(), b.clone()),
            FirNodeType::Wire => Self::Wire,
            FirNodeType::Reg => Self::Reg,
            FirNodeType::RegReset => Self::RegReset,
            FirNodeType::SMem(x) => Self::SMem(x.clone()),
            FirNodeType::CMem => Self::CMem,
            FirNodeType::WriteMemPort(cond) => Self::WriteMemPort(cond.clone()),
            FirNodeType::ReadMemPort(cond) => Self::ReadMemPort(cond.clone()),
            FirNodeType::InferMemPort(cond) => Self::InferMemPort(cond.clone()),
            FirNodeType::Inst(x) => Self::Inst(x.clone()),
            FirNodeType::Input => Self::Input,
            FirNodeType::Output => Self::Output,
            FirNodeType::Phi => Self::Phi,
        }
    }
}

#[derive(Derivative, Clone, PartialEq, Eq, Hash)]
#[derivative(Debug)]
pub struct RippleNodeData {
    pub name: Option<Identifier>,
    pub tpe: RippleNodeType,
    pub tg: GroundType
}

impl RippleNodeData {
    pub fn new(name: Option<Identifier>, tpe: RippleNodeType, tg: GroundType) -> Self {
        Self { name, tpe, tg }
    }
}

impl_clean_display!(RippleNodeData);

define_index_type!(RippleNodeIndex);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RippleNode {
    pub data: RippleNodeData,
    pub id: RippleNodeIndex,
}

impl RippleNode {
    pub fn new(data: RippleNodeData, id: RippleNodeIndex) -> Self {
        Self { data, id }
    }
}

impl_clean_display!(RippleNode);
