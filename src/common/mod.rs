pub mod graphviz;

use pdfium_render::prelude::PdfiumError;
use thiserror::Error;

use crate::ir::firir::FirNode;

#[derive(Debug, Error)]
pub enum RippleIRErr {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("PdfiumError error: {0}")]
    PdfiumError(#[from] PdfiumError),

    #[error("FirNodeError: {0} {1}")]
    FirNodeError(String, FirNode),
}
