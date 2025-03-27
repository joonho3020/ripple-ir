use chirrtl_parser::ast::*;
use indexmap::IndexMap;
use crate::ir::rir::rgraph::*;
use crate::ir::hierarchy::*;

#[derive(Debug, Clone)]
pub struct RippleIR {
    pub name: Identifier,
    pub graphs: IndexMap<Identifier, RippleGraph>,
    pub hierarchy: Hierarchy,
}

impl RippleIR {
    pub fn new(name: Identifier) -> Self {
        Self {
            name,
            graphs: IndexMap::new(),
            hierarchy: Hierarchy::default(),
        }
    }
}
