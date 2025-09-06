//! Some helpers for the application.

use std::path::{Path, PathBuf};

use ftlog::{
    LevelFilter, LoggerGuard,
    appender::{FileAppender, Period},
};
// use rayon::prelude::*;

// /// Compute the inner product of all the given vectors.
// pub fn precompute_ips(vectors: Vec<Vec<f32>>) -> Vec<(f32, Vec<f32>)> {
//     vectors
//         .into_par_iter()
//         .map(|v| (distances::blas::dot_f32(&v, &v), v))
//         .collect()
// }

// /// Compute the euclidean distance squared between two vectors given their precomputed inner products.
// pub fn euc_by_ip((aa, a): &(f32, Vec<f32>), (bb, b): &(f32, Vec<f32>)) -> f32 {
//     // 2_f32.mul_add(-distances::blas::dot_f32(a, b), aa + bb).sqrt()
//     distances::simd::euclidean_f32(a, b)
// }

// /// Compute the cosine distance between two vectors given their precomputed inner products.
// pub fn cos_by_ip((aa, a): &(f32, Vec<f32>), (bb, b): &(f32, Vec<f32>)) -> f32 {
//     distances::blas::dot_f32(a, b).mul_add(-(aa * bb).inv_sqrt(), 1.0)
// }

/// Configures the logger.
///
/// # Errors
///
/// - If a logs directory could not be located/created.
/// - If the logger could not be initialized.
pub fn configure_logger<P: AsRef<Path>>(file_name: &str, logs_dir: &P) -> Result<(LoggerGuard, PathBuf), String> {
    let log_path = logs_dir.as_ref().join(format!("{file_name}.log"));

    let writer = FileAppender::builder().path(&log_path).rotate(Period::Day).build();

    let err_path = log_path.with_extension("err.log");

    let guard = ftlog::Builder::new()
        // global max log level
        .max_log_level(LevelFilter::Trace)
        // define root appender, pass None would write to stderr
        .root(writer)
        // write `Warn` and `Error` logs in ftlog::appender to `err_path` instead of `log_path`
        .filter("ftlog::appender", "ftlog-appender", LevelFilter::Warn)
        .appender("ftlog-appender", FileAppender::new(err_path))
        .try_init()
        .map_err(|e| e.to_string())?;

    Ok((guard, log_path))
}
