use chirrtl_parser::ast::Width;
use crate::ir::fir::FirEdgeType;
use crate::impl_clean_display;
use crate::define_index_type;
use crate::ir::whentree::PrioritizedCond;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RippleEdgeType {
    Wire,

    Operand0,
    Operand1,

    MuxCond,
    MuxTrue,
    MuxFalse,

    Clock,
    Reset,
    DontCare,

    PhiInput(PrioritizedCond),
    PhiSel,
    PhiOut,

    MemPortEdge,
    MemPortAddr,
    MemPortEn,

    ArrayAddr,
}

impl From<&FirEdgeType> for RippleEdgeType {
    fn from(value: &FirEdgeType) -> Self {
        match value {
            FirEdgeType::Wire => RippleEdgeType::Wire,
            FirEdgeType::Operand0 => RippleEdgeType::Operand0,
            FirEdgeType::Operand1 => RippleEdgeType::Operand1,
            FirEdgeType::MuxCond => RippleEdgeType::MuxCond,
            FirEdgeType::MuxTrue => RippleEdgeType::MuxTrue,
            FirEdgeType::MuxFalse => RippleEdgeType::MuxFalse,
            FirEdgeType::Clock => RippleEdgeType::Clock,
            FirEdgeType::Reset => RippleEdgeType::Reset,
            FirEdgeType::DontCare => RippleEdgeType::DontCare,
            FirEdgeType::PhiInput(prior_cond) => RippleEdgeType::PhiInput(prior_cond.clone()),
            FirEdgeType::PhiSel => RippleEdgeType::PhiSel,
            FirEdgeType::PhiOut => RippleEdgeType::PhiOut,
            FirEdgeType::MemPortEdge => RippleEdgeType::MemPortEdge,
            FirEdgeType::MemPortAddr => RippleEdgeType::MemPortAddr,
            FirEdgeType::MemPortEn => RippleEdgeType::MemPortEn,
            FirEdgeType::ArrayAddr => RippleEdgeType::ArrayAddr
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RippleEdgeData {
    pub width: Option<Width>,
    pub et: RippleEdgeType
}

impl RippleEdgeData {
    pub fn new(width: Option<Width>, et: RippleEdgeType) -> Self {
        Self { width, et }
    }
}

impl_clean_display!(RippleEdgeData);

define_index_type!(RippleEdgeIndex);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RippleEdge {
    pub data: RippleEdgeData,
    pub id: RippleEdgeIndex,
}

impl RippleEdge {
    pub fn new(data: RippleEdgeData, id: RippleEdgeIndex) -> Self {
        Self { data, id }
    }
}

impl_clean_display!(RippleEdge);
