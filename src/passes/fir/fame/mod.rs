pub mod fame1;
pub mod fame5;

use crate::ir::typetree::typetree::TypeTree;
use crate::ir::typetree::tnode::GroundType;
use rusty_firrtl::Width;

pub fn log2_ceil(n: u32) -> u32 {
    n.next_power_of_two().trailing_zeros()
}

pub fn uint_ttree(w: u32) -> TypeTree {
    TypeTree::build_from_ground_type(GroundType::UInt(Some(Width(w))))
}
