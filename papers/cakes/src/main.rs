//! Reproducible benchmarks for the CAKES search algorithms.

use std::path::PathBuf;

use abd_clam::{
    Cluster,
    cakes::selection,
    cluster::{BranchingFactor, PartitionStrategy, SpanReductionFactor},
};
use clap::Parser;
use rand::prelude::*;

mod data;
mod utils;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The path to the base directory containing the ANN-Benchmarks datasets.
    #[arg(short('i'), long)]
    inp_dir: PathBuf,

    /// The output directory for the benchmark results. If not provided, results will be written to a `cakes-benchmarks` subdirectory in the input directory.
    #[arg(short('o'), long, default_value = None)]
    out_dir: Option<PathBuf>,

    /// The directory to store log files. If not provided, a `logs` subdirectory in the output directory will be used.
    #[arg(short('l'), long, default_value = None)]
    log_dir: Option<PathBuf>,

    /// The random seed to use.
    #[arg(short('s'), long, default_value = None)]
    seed: Option<u64>,

    /// The number of neighbors to search for.
    #[arg(short('k'), long, default_value = "10")]
    k: usize,

    /// The number of queries to use for algorithm selection. These will be randomly chosen from the training set.
    #[arg(short('q'), long, default_value = "100")]
    q: usize,

    /// The minimum time in seconds to run each throughput measurement during algorithm selection.
    #[arg(short('t'), long, default_value = "5.0")]
    selection_time: f64,

    /// The minimum time in seconds to run the final throughput measurement.
    #[arg(short('T'), long, default_value = "10.0")]
    measurement_time: f64,
}

/// A placeholder main function.
#[allow(
    clippy::cast_precision_loss,
    clippy::while_float,
    clippy::too_many_lines,
    clippy::cognitive_complexity
)]
fn main() -> Result<(), String> {
    let args = Args::parse();

    let inp_dir = args.inp_dir;
    if !inp_dir.exists() {
        return Err(format!("Input directory '{}' does not exist.", inp_dir.display()));
    }
    if !inp_dir.is_dir() {
        return Err(format!("Input path '{}' is not a directory.", inp_dir.display()));
    }

    let out_dir = if let Some(dir) = args.out_dir {
        if !dir.exists() {
            std::fs::create_dir_all(&dir)
                .map_err(|e| format!("Failed to create output directory '{}': {e}", dir.display()))?;
        } else if !dir.is_dir() {
            return Err(format!("Output path '{}' is not a directory.", dir.display()));
        }
        dir
    } else {
        let inp_metadata = std::fs::metadata(&inp_dir).map_err(|e| e.to_string())?;
        if inp_metadata.permissions().readonly() {
            return Err(format!(
                "Input directory '{}' is not writable. Please specify an output directory using the -o option.",
                inp_dir.display()
            ));
        }
        let out_dir = inp_dir.join("cakes-benchmarks");
        if !out_dir.exists() {
            std::fs::create_dir_all(&out_dir)
                .map_err(|e| format!("Failed to create output directory '{}': {e}", out_dir.display()))?;
        }
        out_dir
    };

    let logs_dir = if let Some(dir) = args.log_dir {
        if !dir.exists() {
            std::fs::create_dir_all(&dir)
                .map_err(|e| format!("Failed to create log directory '{}': {e}", dir.display()))?;
        } else if !dir.is_dir() {
            return Err(format!("Log path '{}' is not a directory.", dir.display()));
        }
        dir
    } else {
        let logs_dir = out_dir.join("logs");
        if !logs_dir.exists() {
            std::fs::create_dir_all(&logs_dir)
                .map_err(|e| format!("Failed to create log directory '{}': {e}", logs_dir.display()))?;
        }
        logs_dir
    };
    let (_logger_guard, log_path) = utils::configure_logger("cakes-paper", &logs_dir)?;
    println!("Logging to {}", log_path.display());

    let mut rng = args.seed.map(rand::rngs::StdRng::seed_from_u64);
    let subset = data::AnnDataset::all_datasets();

    let strategy = PartitionStrategy::default()
        .with_branching_factor(BranchingFactor::Unbounded)
        .with_span_reduction(SpanReductionFactor::Phi);

    for dataset in data::AnnDataset::all_datasets() {
        if !subset.contains(&dataset) {
            // Just for quicker development iterations.
            continue;
        }
        let metric = dataset.metric_by_ip();

        ftlog::info!("Reading dataset '{}'", dataset.name());
        let items = dataset.read_train(&inp_dir, rng.as_mut()).map_err(|e| e.to_string())?;
        let items = utils::par_precompute_ips(items);
        let items = items.into_iter().enumerate().collect::<Vec<_>>();

        ftlog::info!("Building CAKES index for dataset '{}'", dataset.name());
        let root = Cluster::<_, _, _, ()>::par_new_tree(items, &metric, &strategy)?;

        ftlog::info!("Selecting fastest algorithm for dataset '{}'", dataset.name());
        let (best_alg, expected_throughput) =
            selection::par_select_fastest_algorithm(&root, &metric, args.q, args.k, args.selection_time);
        ftlog::info!("Selected algorithm {best_alg} with expected throughput {expected_throughput:.8} queries/sec");

        let queries = dataset.read_test(&inp_dir, rng.as_mut()).map_err(|e| e.to_string())?;
        let queries = utils::precompute_ips(queries);
        ftlog::info!("Measuring throughput for dataset '{}'", dataset.name());

        let start = std::time::Instant::now();
        let mut total_queries = 0;
        let min_duration = std::time::Duration::from_secs_f64(args.measurement_time);
        while start.elapsed() < min_duration {
            let _results = best_alg.par_batch_search(&root, &metric, &queries);
            total_queries += queries.len();
        }
        let throughput = total_queries as f64 / start.elapsed().as_secs_f64();
        ftlog::info!(
            "Measured throughput for dataset '{}' is {throughput:.8} queries/sec",
            dataset.name()
        );
    }

    Ok(())
}
