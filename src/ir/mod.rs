pub mod typetree;
pub mod whentree;
pub mod firir;
pub mod hierarchy;
pub mod rir;

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct PhiPriority {
    /// Priority between blocks
    /// - Smaller number means higher priority
    pub block: u32,

    /// Priority between statements within the same block
    /// - Smaller number means higher priority
    pub stmt: u32,
}

impl PhiPriority {
    pub fn new(block: u32, stmt: u32) -> Self {
        Self { block, stmt }
    }
}

