use std::fmt::Display;

#[derive(Default, Debug, Clone, Copy)]
pub struct TypeTreeEdge;

impl Display for TypeTreeEdge {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}