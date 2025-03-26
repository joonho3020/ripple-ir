pub mod graphviz;

use pdfium_render::prelude::PdfiumError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RippleIRErr {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("PdfiumError error: {0}")]
    PdfiumError(#[from] PdfiumError),

    #[error("TypeTreeInferenceError: {0}")]
    TypeTreeInferenceError(String),

    #[error("PhiNodeError: {0}")]
    PhiNodeError(String),
}

#[macro_export]
macro_rules! timeit {
    ($label:expr, $block:expr) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let elapsed = start.elapsed();
        println!("{} took: {:?}", $label, elapsed);
        result
    }};
}
