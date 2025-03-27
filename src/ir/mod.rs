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

/// Generates unique indices
#[derive(Debug, Default, Clone)]
pub struct IndexGen {
    next_index: u32,
}

impl IndexGen {
    pub fn new() -> Self {
        Self { next_index: 0 }
    }

    /// Generate a new unique index of type `T` that implements `From<u32>`
    pub fn generate<T>(&mut self) -> T
    where
        T: From<u32>,
    {
        let idx = self.next_index;
        self.next_index += 1;
        T::from(idx)
    }
}
