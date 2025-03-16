pub mod graphviz;

use thiserror::Error;

use crate::ir::firir::FirNode;

#[derive(Debug, Error)]
pub enum RippleIRErr {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("FirNodeError: {0} {1}")]
    FirNodeError(String, FirNode),
}
