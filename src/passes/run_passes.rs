use crate::passes::{remove_unnecessary_phi::*, flatten::*};
use crate::ir::*;

pub fn run_passes(ir: &mut RippleIR) {
    remove_unnecessary_phi(ir);
    flatten(ir);
}
