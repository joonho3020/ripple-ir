use chirrtl_parser::ast::*;
use std::cmp::Ordering;
use crate::define_index_type;
use crate::impl_clean_display;

/// - Direction in the perspective of the noding holding this `TypeTree`
/// ```
/// o <-- Incoming ---
/// o --- Outgoing -->
/// ```
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TypeDirection {
    #[default]
    Outgoing,
    Incoming,
}

impl TypeDirection {
    pub fn flip(&self) -> Self {
        match self {
            Self::Incoming => Self::Outgoing,
            Self::Outgoing => Self::Incoming,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GroundType {
    Invalid,
    DontCare,
    Clock,
    Reset,
    AsyncReset,
    UInt,
    SInt,
    SMem,
    CMem,
    Inst,
}

impl From<&TypeGround> for GroundType {
    fn from(value: &TypeGround) -> Self {
        match value {
            TypeGround::SInt(..) => Self::SInt,
            TypeGround::UInt(..) => Self::UInt,
            TypeGround::Clock => Self::Clock,
            TypeGround::Reset => Self::Reset,
            TypeGround::AsyncReset => Self::AsyncReset,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TypeTreeNodeType {
    Ground(GroundType),
    Fields,
    Array,
}

define_index_type!(TypeTreeNodeIndex);

/// Node in the TypeTree
#[derive(Debug, Clone, Hash, Eq, Ord)]
pub struct TypeTreeNode {
    /// Name of this node. The root of the tree will not have a name
    pub name: Option<Identifier>,

    /// Direction of this node
    pub dir: TypeDirection,

    /// NodeType
    pub tpe: TypeTreeNodeType,

    /// Unique index of this node
    pub id: Option<TypeTreeNodeIndex>
}

impl TypeTreeNode {
    pub fn new(name: Option<Identifier>, dir: TypeDirection, tpe: TypeTreeNodeType) -> Self {
        Self { name, dir, tpe, id: None }
    }
}

impl PartialEq for TypeTreeNode {
    fn eq(&self, other: &Self) -> bool {
        (self.name == other.name) && (self.tpe == other.tpe)
    }
}

impl PartialOrd for TypeTreeNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Compare only dir, tpe, and id — ignore name
        Some((
            &self.name,
            &self.tpe,
        ).cmp(&(
            &other.name,
            &other.tpe,
        )))
    }
}

impl_clean_display!(TypeTreeNode);
