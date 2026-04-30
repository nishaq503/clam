//! Python wrappers for distance functions implemented in Rust.

pub(crate) mod simd;
pub(crate) mod strings;
mod utils;
pub(crate) mod vectors;

use pyo3::prelude::*;

/// The `abd-distances` module implemented in Rust.
#[pymodule]
fn abd_distances(m: &Bound<'_, PyModule>) -> PyResult<()> {
    simd::register(m)?;
    strings::register(m)?;
    vectors::register(m)?;
    Ok(())
}

/*
Might need to export a library path for the package to work on MacOS:

export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/gcc/15.2.0_1/lib/gcc/current:$DYLD_LIBRARY_PATH
*/
