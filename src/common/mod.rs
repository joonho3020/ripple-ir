pub mod graphviz;

use std::string::FromUtf8Error;
use std::process::Command;
use pdfium_render::prelude::PdfiumError;
use thiserror::Error;
use spinoff::{Spinner, spinners};
use clap::ValueEnum;
use rusty_firrtl::Annotations;

#[derive(Debug, Error)]
pub enum RippleIRErr {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("SerDes error: {0}")]
    SerDesJsonError(#[from] serde_json::Error),

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

#[derive(Debug, Clone, ValueEnum)]
pub enum FIRRTLVersion {
    Chirrtl,
    Firrtl3,
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

pub fn run_firtool(firrtl_filename: &str, outdir: &str) -> Result<String, RippleIRErr> {
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
        .arg("--disable-all-randomization")
        .arg("--mlir-timing")
        .arg("--lowering-options=emittedLineLength=2048,noAlwaysComb,disallowLocalVariables,verifLabels,disallowPortDeclSharing,locationInfoStyle=wrapInAtSquareBracket")
        .arg("--split-verilog")
        .arg("-o")
        .arg(outdir)
        .arg(firrtl_filename).output().expect("firtool command to run");

    spinner.success("Finished running firtool");

    let stderr = String::from_utf8(cmd_out.stderr)?;
    if stderr.contains("error: ") {
        Err(RippleIRErr::CIRCTError(stderr))
    } else {
        Ok(String::from_utf8(cmd_out.stdout)?)
    }
}

pub fn export_circuit(fir_file: &str, out_dir: &str) -> Result<(), RippleIRErr> {
    match run_firtool(&fir_file, out_dir) {
        Ok(..) => {
            return Ok(())
        }
        Err(RippleIRErr::CIRCTError(_e)) => {
            return Err(RippleIRErr::MiscError(format!("to_ast failed for module {:?}", fir_file)));
        }
        _ => {
            unreachable!()
        }
    }
}

pub fn read_annos(filepath: &str) -> Result<Annotations, RippleIRErr> {
    let annos_str = std::fs::read_to_string(filepath)?;
    let annos = Annotations{ 0: serde_json::from_str(&annos_str).unwrap() };
    Ok(annos)
}

pub fn write_annos(annos: &Annotations, filepath: &str) -> Result<(), RippleIRErr> {
    let file = std::fs::File::create(filepath)?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &annos.0)?;
    Ok(())
}
