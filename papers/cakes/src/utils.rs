//! Some helpers for the application.

use std::path::{Path, PathBuf};

use ftlog::{
    LevelFilter, LoggerGuard,
    appender::{FileAppender, Period},
};

/// The euclidean distance metric.
pub fn euclidean<I: AsRef<[f32]>>(a: &I, b: &I) -> f32 {
    distances::simd::euclidean_f32(a.as_ref(), b.as_ref())
}

/// The cosine distance function.
pub fn cosine<I: AsRef<[f32]>>(a: &I, b: &I) -> f32 {
    distances::simd::cosine_f32(a.as_ref(), b.as_ref())
}

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
