//! Run Cakes search on `RadioML` data.

use std::{path::Path, time::Instant};

use abd_clam::{Cakes, Dataset, PartitionCriteria, VecDataset};
use distances::Number;
use mt_logger::{mt_level, mt_log, mt_stream, Level, OutputStream};

pub mod metrics;
mod reader;

#[allow(clippy::module_name_repetitions)]
pub use reader::RadioMLMetadata;
use reader::Snr;
use serde::{Deserialize, Serialize};

/// Run Cakes search on `RadioML` data.
#[allow(clippy::too_many_lines)]
pub fn run(
    input_dir: &Path,
    snr: Snr,
    sample_size: usize,
    num_queries: usize,
    ks: &[usize],
    seed: Option<u64>,
    output_dir: &Path,
) -> Result<(), String> {
    mt_stream!(OutputStream::Both);
    mt_level!(Level::Debug);

    // Check that the input directory exists.
    if !input_dir.exists() {
        return Err(format!(
            "The input directory {} does not exist.",
            input_dir.display()
        ));
    }

    // Check that we have a valid SNR.
    if !((-20..30).contains(&snr) && snr % 2 == 0) {
        return Err(format!(
            "The SNR must be one of -20, -18, ..., 28, 30. Got {snr}."
        ));
    }

    // Check that the sample size and number of queries are valid.
    if sample_size + num_queries > 4096 {
        return Err("The sum of sample_size and num_queries must be <= 4096.".to_string());
    }

    // Check that the output directory exists.
    if !output_dir.exists() {
        return Err(format!(
            "The output directory {} does not exist.",
            output_dir.display()
        ));
    }

    mt_log!(
        Level::Info,
        "Reading RadioML data from {}",
        input_dir.display()
    );

    let [train_data, query_data] = reader::read(input_dir, snr, sample_size, num_queries, seed)?;
    assert_eq!(
        train_data.len(),
        sample_size * reader::ModulationMode::variant_count(),
        "Expected {} samples, got {}",
        sample_size * reader::ModulationMode::variant_count(),
        train_data.len()
    );
    let cardinality = train_data.len();

    mt_log!(
        Level::Info,
        "Read {} samples",
        train_data.len() + query_data.len()
    );

    // Create the path to the model
    let cakes_dir = output_dir.join(format!(
        "cakes-snr-{}-cardinality-{}",
        snr + 20,
        train_data.len()
    ));

    // If the output directory already exists, load CAKES from it.
    let (cakes, build_time) = if cakes_dir.exists() {
        mt_log!(Level::Info, "Loading CAKES from {}", cakes_dir.display());
        let cakes = Cakes::load(&cakes_dir, metrics::dtw, true)?;

        let tuned_algorithm = cakes.tuned_knn_algorithm();
        mt_log!(
            Level::Info,
            "Tuned knn algorithm was: {}",
            tuned_algorithm.name()
        );

        (cakes, 0.0)
    } else {
        let (train_metadata, train) = train_data.into_iter().unzip();

        let data_name = "RadioML".to_string();
        let metric = metrics::dtw;
        let is_expensive = true;

        let train = VecDataset::new(data_name, train, metric, is_expensive)
            .assign_metadata(train_metadata)?;
        let max_depth = 16 * train.cardinality().ilog2() as usize;
        let criteria = PartitionCriteria::new(true).with_max_depth(max_depth);

        mt_log!(Level::Info, "Creating search tree to max depth {max_depth}");
        let start = Instant::now();
        let cakes = Cakes::new(train, seed, &criteria);
        let build_time = start.elapsed().as_secs_f32();
        mt_log!(
            Level::Info,
            "Created search tree in {build_time:.2e} seconds."
        );

        mt_log!(Level::Info, "Saving CAKES to {}", cakes_dir.display());
        std::fs::create_dir_all(&cakes_dir).map_err(|e| e.to_string())?;
        cakes.save(&cakes_dir)?;

        (cakes, build_time)
    };

    let (_, queries): (Vec<_>, Vec<_>) = query_data.into_iter().unzip();
    let queries = queries.iter().collect::<Vec<_>>();

    // Run knn-search in batches of 10 queries for up-to 10 seconds.
    let batch_size = 100;

    for &k in ks {
        mt_log!(Level::Info, "Running knn search with k = {}", k);

        for algo in abd_clam::knn::Algorithm::variants() {
            let mut num_queries_run = 0;
            let start = Instant::now();
            for batch in queries.chunks(batch_size) {
                let results =
                    cakes.batch_knn_search(batch, k, abd_clam::knn::Algorithm::GreedySieve);
                num_queries_run += batch.len();
                mt_log!(
                    Level::Trace,
                    "Ran knn search on {num_queries_run} queries so far. Got {} results.",
                    results.len()
                );

                if start.elapsed().as_secs_f32() > 10.0 {
                    break;
                }
            }
            let search_time = start.elapsed().as_secs_f32();
            let throughput = num_queries_run.as_f32() / search_time;

            mt_log!(
                Level::Info,
                "Algorithm {} had throughput {throughput:.2e} QPS.",
                algo.name()
            );

            // Make a report
            Report {
                dataset: "radioml",
                metric: "dtw",
                cardinality,
                built: true,
                build_time,
                shard_sizes: cakes.shard_cardinalities(),
                num_queries: num_queries_run,
                k,
                algorithm: algo.name(),
                throughput,
            }
            .save(output_dir)?;
        }
    }

    Ok(())
}

/// A report of the results of an ANN benchmark.
#[derive(Debug, Serialize, Deserialize)]
struct Report<'a> {
    /// Name of the data set.
    dataset: &'a str,
    /// Name of the distance function.
    metric: &'a str,
    /// Number of data points in the data set.
    cardinality: usize,
    /// Whether the search tree was built or loaded from disk.
    built: bool,
    /// Time taken to build the search tree or load it from disk.
    build_time: f32,
    /// Sizes of the shards created for `ShardedCakes`.
    shard_sizes: Vec<usize>,
    /// Number of queries used for search.
    num_queries: usize,
    /// Value of k used for knn-search.
    k: usize,
    /// Name of the algorithm used.
    algorithm: &'a str,
    /// Throughput of the tuned algorithm.
    throughput: f32,
}

impl Report<'_> {
    /// Save the report to a file in the given directory.
    ///
    /// # Arguments
    ///
    /// * `dir`: The directory to save the report to.
    ///
    /// # Errors
    ///
    /// * If the directory does not exist or cannot be written to.
    /// * If the report cannot be serialized.
    fn save(&self, dir: &Path) -> Result<(), String> {
        if !dir.exists() {
            return Err(format!("Directory {dir:?} does not exist."));
        }

        if !dir.is_dir() {
            return Err(format!("{dir:?} is not a directory."));
        }

        let path = dir.join(format!(
            "{}-{}-{}-knn-{}.json",
            self.dataset, self.cardinality, self.algorithm, self.k
        ));
        let report = serde_json::to_string_pretty(&self).map_err(|e| e.to_string())?;
        std::fs::write(path, report).map_err(|e| e.to_string())?;
        Ok(())
    }
}
