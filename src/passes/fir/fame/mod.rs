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


#[cfg(test)]
pub mod test {
    use crate::passes::fir::fame::log2_ceil;

    #[test]
    fn log2_ceil_2() {
        assert_eq!(log2_ceil(2), 1);
    }

    #[test]
    fn log2_ceil_3() {
        assert_eq!(log2_ceil(3), 2);
    }
}
