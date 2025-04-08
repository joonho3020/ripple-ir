pub mod graphviz;

use std::string::FromUtf8Error;
use std::process::Command;
use pdfium_render::prelude::PdfiumError;
use thiserror::Error;
use spinoff::{Spinner, spinners};

#[derive(Debug, Error)]
pub enum RippleIRErr {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("Failed to convert from utf8 to string: {0}")]
    FromUtf8Error(#[from] FromUtf8Error),

    #[error("PdfiumError error: {0}")]
    PdfiumError(#[from] PdfiumError),

    #[error("TypeTreeInferenceError: {0}")]
    TypeTreeInferenceError(String),

    #[error("PhiNodeError: {0}")]
    PhiNodeError(String),

    #[error("Output FIRRTL file is incompatible with CIRCT: {0}")]
    CIRCTError(String),

    #[error("Misc: {0}")]
    MiscError(String),
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

pub fn run_firtool(firrtl_filename: &str) -> Result<String, RippleIRErr> {
    let mut spinner = Spinner::new(
        spinners::Dots,
        format!("Running firtool on {}...", firrtl_filename),
        None);

    Command::new("which")
        .arg("firtool")
        .output().expect("to succeed");

    let cmd_out = Command::new("firtool")
        .arg("--format=fir")
        .arg("--export-module-hierarchy")
        .arg("--verify-each=true")
        .arg("--warn-on-unprocessed-annotations")
        .arg("--disable-annotation-classless")
        .arg("--disable-annotation-unknown")
        .arg("--mlir-timing")
        .arg("--lowering-options=emittedLineLength=2048,noAlwaysComb,disallowLocalVariables,verifLabels,disallowPortDeclSharing,locationInfoStyle=wrapInAtSquareBracket")
        .arg("--split-verilog")
        .arg("-o")
        .arg("test-outputs/verilog")
        .arg(firrtl_filename).output().expect("firtool command to run");

    spinner.success("Finished running firtool");

    let stderr = String::from_utf8(cmd_out.stderr)?;
    if stderr.contains("error: ") {
        Err(RippleIRErr::CIRCTError(stderr))
    } else {
        Ok(String::from_utf8(cmd_out.stdout)?)
    }
}
