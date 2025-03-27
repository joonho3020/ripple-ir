pub mod agg;
pub mod rir;
pub mod redge;
pub mod rnode;
pub mod rgraph;

#[macro_export]
macro_rules! define_index_type {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u32);

        impl $name {
            pub fn to_usize(&self) -> usize {
                self.0 as usize
            }
        }

        impl From<u32> for $name {
            fn from(value: u32) -> Self {
                $name(value)
            }
        }

        impl Into<u32> for $name {
            fn into(self) -> u32 {
                self.0 as u32
            }
        }

        impl From<usize> for $name {
            fn from(value: usize) -> Self {
                Self(value as u32)
            }
        }

        impl From<$name> for usize {
            fn from(value: $name) -> Self {
                value.0 as usize
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }
    };
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
