#![deny(clippy::correctness)]
#![warn(
    missing_docs,
    clippy::all,
    clippy::suspicious,
    clippy::style,
    clippy::complexity,
    clippy::perf,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]

//! CLI for running Cakes experiments and benchmarks.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use mt_logger::{mt_flush, mt_log, mt_new, Level, OutputStream};

mod genomic;
mod lfd_plots;
mod radio;
mod utils;
mod vectors;

/// CLI for running Cakes experiments and benchmarks.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the directory with the data sets. The directory should contain
    /// the hdf5 files downloaded from the ann-benchmarks repository.
    #[arg(long)]
    input_dir: PathBuf,

    /// Output directory for the report.
    #[arg(long)]
    output_dir: PathBuf,

    /// Optional seed for the random number generator.
    #[arg(long)]
    seed: Option<u64>,

    /// Subcommands for the CLI.
    #[command(subcommand)]
    command: Commands,
}

/// Subcommands for the CLI.
#[derive(Subcommand)]
enum Commands {
    /// Runs RNN search.
    Rnn {
        /// Name of the data set to use.
        #[arg(long)]
        dataset: String,

        /// Whether to shard the data set for search.
        #[arg(long)]
        use_shards: bool,

        /// The maximum time (in seconds) to use for running search.
        #[arg(long, default_value = "10")]
        max_search_time: f32,

        /// The root radius will be divided by these values to get the radii
        /// for search.
        #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "100 20")]
        radius_divisors: Vec<u32>,
    },

    /// Runs KNN search.
    Knn {
        /// Name of the data set to use.
        #[arg(long)]
        dataset: String,

        /// Whether to shard the data set for search.
        #[arg(long)]
        use_shards: bool,

        /// The depth of the tree to use for auto-tuning knn-search.
        #[arg(long, default_value = "10")]
        tuning_depth: usize,

        /// The value of k to use for auto-tuning knn-search.
        #[arg(long, default_value = "10")]
        tuning_k: usize,

        /// The maximum time (in seconds) to use for running search.
        #[arg(long, default_value = "10")]
        max_search_time: f32,

        /// Number of nearest neighbors to search for.
        #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "10 100")]
        ks: Vec<usize>,
    },

    /// Generates augmented data sets for scaling experiments.
    Scaling {
        /// Name of the data set to use.
        #[arg(long)]
        dataset: String,

        /// Maximum scaling factor. The data set will be scaled by factors of
        /// `2 ^ i` for `i` in `0..=max_scale`.
        #[arg(long, default_value = "16")]
        max_scale: u32,

        /// Error rate used for scaling.
        #[arg(long, default_value = "0.01")]
        error_rate: f32,

        /// Maximum memory usage (in gigabytes) for the scaled data sets.
        #[arg(long, default_value = "256")]
        max_memory: f32,

        /// Whether to overwrite existing augmented data sets.
        #[arg(long)]
        overwrite: bool,

        /// The maximum time (in seconds) to use for running search.
        #[arg(long, default_value = "10")]
        max_search_time: f32,

        /// Number of nearest neighbors to search for.
        #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "10 100")]
        ks: Vec<usize>,

        /// The root radius will be divided by these values to get the radii
        /// for search.
        #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "100 20")]
        radius_divisors: Vec<u32>,
    },

    /// Runs KNN and RNN search on Silva-18S dataset.
    Genomic {
        /// The metric to use for computing distances. One of "hamming",
        /// "levenshtein", or "needleman-wunsch".
        #[arg(long, default_value = "levenshtein")]
        metric: String,

        /// The number of sequences to sample from the data set for building
        /// the index. The number of queries will be 10% of this, up to 1000.
        #[arg(long)]
        sample_size: Option<usize>,

        /// The depth of the tree to use for auto-tuning knn-search.
        #[arg(long, default_value = "6")]
        tuning_depth: usize,

        /// The value of k to use for auto-tuning knn-search.
        #[arg(long, default_value = "10")]
        tuning_k: usize,

        /// Number of nearest neighbors to search for.
        #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "10 100")]
        ks: Vec<usize>,

        /// Radii to use for range search.
        #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "25 100 250")]
        rs: Vec<usize>,
    },

    /// Runs majority voting via knn on RadioML data.
    Radio {
        /// The signal-to-noise ratio to for sampling from the data set for
        /// building the index. If not specified, all SNRs will be used.
        ///
        /// This must be an even number between -20 and 30 (inclusive).
        #[arg(long, default_value = "10")]
        snr: i32,

        /// How many samples at each SNR and modulation mode to use for
        /// building the index. If not specified, all samples will be used.
        ///
        /// The sum of `sample_size` and `num_queries` must be <= 4096.
        #[arg(long, default_value = "128")]
        sample_size: usize,

        /// The number of queries to use for each modulation mode. If not
        /// specified, 10 queries will be used for each modulation mode.
        ///
        /// The sum of `sample_size` and `num_queries` must be <= 4096.
        #[arg(long, default_value = "10")]
        num_queries: usize,

        /// Values of k to use for majority voting.
        #[arg(long, value_parser, num_args = 1.., value_delimiter = ' ', default_value = "5 25 51")]
        ks: Vec<usize>,
    },

    /// Export cluster properties as a CSV file for plotting.
    LfdPlots {
        /// Name of the data set to use.
        #[arg(long)]
        dataset: String,
    },
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), String> {
    std::env::set_var("RUST_BACKTRACE", "full");

    mt_new!(None, Level::Info, OutputStream::StdOut);

    let cli = Cli::parse();

    check_dir(&cli.input_dir)?;
    check_dir(&cli.output_dir)?;

    mt_log!(Level::Info, "Input directory: {}", cli.input_dir.display());
    mt_log!(
        Level::Info,
        "Output directory: {}",
        cli.output_dir.display()
    );

    // You can check for the existence of subcommands, and if found use their
    // matches just as you would the top level cmd
    match &cli.command {
        Commands::Rnn {
            dataset,
            use_shards,
            radius_divisors,
            max_search_time,
        } => {
            vectors::rnn_search(
                &cli.input_dir,
                dataset,
                *use_shards,
                *max_search_time,
                radius_divisors,
                cli.seed,
                &cli.output_dir,
            )?;
        }
        Commands::Knn {
            dataset,
            use_shards,
            tuning_depth,
            tuning_k,
            max_search_time,
            ks,
        } => {
            vectors::knn_search(
                &cli.input_dir,
                dataset,
                *use_shards,
                Some((*tuning_depth, *tuning_k)),
                *max_search_time,
                ks,
                cli.seed,
                &cli.output_dir,
            )?;
        }
        Commands::Scaling {
            dataset,
            max_scale,
            error_rate,
            max_memory,
            max_search_time,
            overwrite,
            ks,
            radius_divisors,
        } => {
            let scaled_names = vectors::augment_dataset(
                &cli.input_dir,
                dataset,
                *max_scale,
                *error_rate,
                *max_memory,
                *overwrite,
                &cli.input_dir,
            )?;

            for data_name in scaled_names {
                vectors::rnn_search(
                    &cli.input_dir,
                    &data_name,
                    false,
                    *max_search_time,
                    radius_divisors,
                    cli.seed,
                    &cli.output_dir,
                )?;
                vectors::knn_search(
                    &cli.input_dir,
                    &data_name,
                    false,
                    None,
                    *max_search_time,
                    ks,
                    cli.seed,
                    &cli.output_dir,
                )?;
            }
        }
        Commands::Genomic {
            sample_size,
            metric,
            tuning_depth,
            tuning_k,
            ks,
            rs,
        } => {
            genomic::run(
                &cli.input_dir,
                metric,
                *sample_size,
                cli.seed,
                *tuning_depth,
                *tuning_k,
                ks,
                rs,
                &cli.output_dir,
            )?;
        }
        Commands::Radio {
            snr,
            sample_size,
            num_queries,
            ks,
        } => {
            radio::run(
                &cli.input_dir,
                *snr,
                *sample_size,
                *num_queries,
                ks,
                cli.seed,
                &cli.output_dir,
            )?;
        }
        Commands::LfdPlots { dataset } => {
            lfd_plots::save_csv(dataset, &cli.input_dir, &cli.output_dir)?;
        }
    }

    mt_flush!().map_err(|e| e.to_string())?;

    Ok(())
}

/// Checks that the given path exists and is a directory.
///
/// # Arguments
///
/// * `path` - The path to check.
///
/// # Errors
///
/// * If the path does not exist.
/// * If the path is not a directory.
fn check_dir(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!(
            "The input directory '{}' does not exist.",
            path.display()
        ));
    }
    if !path.is_dir() {
        return Err(format!(
            "The input directory '{}' is not a directory.",
            path.display()
        ));
    }
    Ok(())
}
