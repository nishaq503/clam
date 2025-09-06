//! Reproducible benchmarks for the Musals search algorithms.

use std::path::PathBuf;

use abd_clam::{
    PartitionStrategy, Tree,
    cakes::{KnnBfs, KnnDfs, ParSearch, selection},
    partition_strategy::{BranchingFactor, SpanReductionFactor},
};
use clap::Parser;
use rand::prelude::*;

mod data;
mod utils;

/// CLI arguments for the benchmark program.
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
#[allow(clippy::cast_precision_loss, clippy::while_float, clippy::too_many_lines, clippy::cognitive_complexity)]
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
            std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create output directory '{}': {e}", dir.display()))?;
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
            std::fs::create_dir_all(&out_dir).map_err(|e| format!("Failed to create output directory '{}': {e}", out_dir.display()))?;
        }
        out_dir
    };

    let logs_dir = if let Some(dir) = args.log_dir {
        if !dir.exists() {
            std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create log directory '{}': {e}", dir.display()))?;
        } else if !dir.is_dir() {
            return Err(format!("Log path '{}' is not a directory.", dir.display()));
        }
        dir
    } else {
        let logs_dir = out_dir.join("logs");
        if !logs_dir.exists() {
            std::fs::create_dir_all(&logs_dir).map_err(|e| format!("Failed to create log directory '{}': {e}", logs_dir.display()))?;
        }
        logs_dir
    };
    let (_logger_guard, log_path) = utils::configure_logger("cakes-paper", &logs_dir)?;
    println!("Logging to {}", log_path.display());

    let mut rng = args.seed.map(rand::rngs::StdRng::seed_from_u64);
    let subset = data::AnnDataset::all_datasets();
    // let subset = [data::AnnDataset::FashionMnist];

    let strategies = vec![
        PartitionStrategy::default().with_branching_factor(BranchingFactor::Fixed(2)),
        PartitionStrategy::default().with_branching_factor(BranchingFactor::Logarithmic),
        PartitionStrategy::default().with_branching_factor(BranchingFactor::Adaptive(128)),
        PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Sqrt2),
        PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Two),
        PartitionStrategy::default().with_span_reduction(SpanReductionFactor::E),
        PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Pi),
        PartitionStrategy::default().with_span_reduction(SpanReductionFactor::Phi),
    ];

    let bench_search = false;

    for dataset in data::AnnDataset::all_datasets() {
        if !subset.contains(&dataset) {
            // Just for quicker development iterations.
            continue;
        }
        let data_out_dir = out_dir.join(dataset.file_name_prefix());
        if !data_out_dir.exists() {
            std::fs::create_dir_all(&data_out_dir).map_err(|e| format!("Failed to create data output directory '{}': {e}", data_out_dir.display()))?;
        }

        let metric = dataset.metric();

        ftlog::info!("Reading dataset '{}'", dataset.name());
        let mut items = dataset
            .read_train(&inp_dir, rng.as_mut())
            .map_err(|e| e.to_string())?
            .into_iter()
            .enumerate()
            .collect();
        // let mut items = utils::precompute_ips(items).into_iter().enumerate().collect();

        let algorithms: Vec<Box<dyn ParSearch<_, _, _, (), _>>> = vec![
            Box::new(KnnDfs(args.k)),
            Box::new(KnnBfs(args.k)),
            // Box::new(KnnRrnn(k)),
            // Box::new(KnnBranch(k)),
        ];
        let algorithms = algorithms.iter().map(|alg| alg.as_ref()).collect::<Vec<_>>();
        // let algorithms = algorithms.as_slice();

        for strategy in &strategies {
            let strategy = strategy.with_radius_greater_than(1e-6);
            ftlog::info!("Building CAKES index for dataset '{}' with strategy {strategy}", dataset.name());
            let tree = Tree::par_new(items, metric, &strategy, &|_| (), 128)?;

            let root_csv_path = data_out_dir.join(strategy.to_string().to_ascii_lowercase() + "-tree.csv");
            if root_csv_path.exists() {
                std::fs::remove_file(&root_csv_path).map_err(|e| format!("Failed to remove existing file '{}': {e}", root_csv_path.display()))?;
            }
            tree.root().to_csv(&root_csv_path).map_err(|e| e.to_string())?;
            ftlog::info!("Wrote cluster tree to '{}'", root_csv_path.display());

            if bench_search {
                let queries = dataset.read_test(&inp_dir, rng.as_mut()).map_err(|e| e.to_string())?;
                // let queries = utils::precompute_ips(queries);

                ftlog::info!("Selecting fastest algorithm for dataset '{}'", dataset.name());
                let (best_alg, expected_throughput) = selection::par_select_fastest_algorithm(&tree, args.q, args.selection_time, algorithms.as_slice());
                ftlog::info!(
                    "Selected algorithm {} with expected throughput {expected_throughput:.8} queries/sec",
                    best_alg.name()
                );

                ftlog::info!("Measuring throughput for dataset '{}'", dataset.name());
                let mut results = Vec::new();
                let mut total_queries = 0;

                let min_duration = std::time::Duration::from_secs_f64(args.measurement_time);
                let start = std::time::Instant::now();
                while start.elapsed() < min_duration {
                    results = best_alg.par_batch_search(&tree, &queries);
                    total_queries += queries.len();
                }
                let time_taken = start.elapsed().as_secs_f64();
                let throughput = total_queries as f64 / time_taken;
                ftlog::info!("Measured throughput for dataset '{}' is {throughput:.8} queries/sec", dataset.name());

                ftlog::info!("Writing report...");

                let (neighbors, distances): (Vec<Vec<u64>>, Vec<Vec<f32>>) =
                    results.into_iter().map(|row| row.into_iter().map(|(i, d)| (i as u64, d)).unzip()).unzip();
                let neighbors_path = data_out_dir.join(strategy.to_string().to_ascii_lowercase() + &best_alg.name() + "-neighbors.npy");
                if neighbors_path.exists() {
                    std::fs::remove_file(&neighbors_path).map_err(|e| format!("Failed to remove existing file '{}': {e}", neighbors_path.display()))?;
                }
                let neighbors_arr = nested_vec_to_arr(neighbors)?;
                ndarray_npy::write_npy(&neighbors_path, &neighbors_arr).map_err(|e| e.to_string())?;
                ftlog::info!("Wrote neighbors to '{}'", neighbors_path.display());

                let distances_path = data_out_dir.join(strategy.to_string().to_ascii_lowercase() + &best_alg.name() + "-distances.npy");
                if distances_path.exists() {
                    std::fs::remove_file(&distances_path).map_err(|e| format!("Failed to remove existing file '{}': {e}", distances_path.display()))?;
                }
                let distances_arr = nested_vec_to_arr(distances)?;
                ndarray_npy::write_npy(&distances_path, &distances_arr).map_err(|e| e.to_string())?;
                ftlog::info!("Wrote distances to '{}'", distances_path.display());

                let performance_path = data_out_dir.join(strategy.to_string().to_ascii_lowercase() + &best_alg.name() + "-performance.json");
                if performance_path.exists() {
                    std::fs::remove_file(&performance_path).map_err(|e| format!("Failed to remove existing file '{}': {e}", performance_path.display()))?;
                }
                let performance = serde_json::json!({
                    "num_queries": queries.len(),
                    "total_time_seconds": time_taken,
                    "throughput": throughput,
                    "recall": 1.0,  // TODO: compute actual recall
                    "rd_err": 0.0,    // TODO: compute actual relative distance error
                });
                std::fs::write(&performance_path, serde_json::to_string_pretty(&performance).map_err(|e| e.to_string())?)
                    .map_err(|e| format!("Failed to write performance file '{}': {e}", performance_path.display()))?;
                ftlog::info!("Wrote performance report to '{}'", performance_path.display());
            }

            items = tree.take_items();
        }
    }

    Ok(())
}

/// Converts a nested vector into a 2D ndarray Array.
///
/// # Errors
///
/// - If the nested vectors do not form a proper rectangular shape.
fn nested_vec_to_arr<T: ndarray_npy::WritableElement>(nested_vec: Vec<Vec<T>>) -> Result<ndarray::Array2<T>, String> {
    let rows = nested_vec.len();
    let cols = nested_vec.first().map_or(0, Vec::len);
    let flat = nested_vec.into_iter().flatten().collect();
    ndarray::Array2::from_shape_vec((rows, cols), flat).map_err(|e| e.to_string())
}
