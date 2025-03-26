use chirrtl_parser::ast::*;
use petgraph::graph::NodeIndex;
use crate::ir::*;

/// Cleanup
/// - Array and memory nodes
/// - Remove phi selection signals that are irrelevant
pub fn cleanup_rir(rir: &mut RippleIR) {
}
