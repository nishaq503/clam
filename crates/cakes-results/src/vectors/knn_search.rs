//! Benchmark the performance of KNN search algorithms.

use std::{path::Path, time::Instant};

use abd_clam::{knn, Cakes, Dataset, PartitionCriteria, VecDataset};
use distances::Number;
use mt_logger::{mt_log, Level};
use num_format::ToFormattedString;
use serde::{Deserialize, Serialize};

use crate::{utils::format_f32, vectors::ann_datasets::AnnDatasets};

/// Report the results of an ANN benchmark.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn knn_search(
    input_dir: &Path,
    dataset: &str,
    use_shards: bool,
    tuning: Option<(usize, usize)>,
    max_search_time: f32,
    ks: &[usize],
    seed: Option<u64>,
    output_dir: &Path,
) -> Result<(), String> {
    mt_log!(Level::Info, "Start knn_search on {dataset}");

    let dataset = AnnDatasets::from_str(dataset)?;
    let metric = dataset.metric()?;
    let [train_data, queries] = dataset.read(input_dir)?;
    mt_log!(Level::Info, "Dataset: {}", dataset.name());

    let (cardinality, dimensionality) = (train_data.len(), train_data[0].len());
    mt_log!(
        Level::Info,
        "Cardinality: {}",
        cardinality.to_formatted_string(&num_format::Locale::en)
    );
    mt_log!(
        Level::Info,
        "Dimensionality: {}",
        dimensionality.to_formatted_string(&num_format::Locale::en)
    );

    let queries = queries.iter().collect::<Vec<_>>();
    let num_queries = queries.len();
    mt_log!(
        Level::Info,
        "Number of queries: {}",
        num_queries.to_formatted_string(&num_format::Locale::en)
    );

    let (mut cakes, tree_build_time) = if use_shards {
        let max_cardinality = if cardinality < 1_000_000 {
            cardinality
        } else if cardinality < 5_000_000 {
            100_000
        } else {
            1_000_000
        };

        let shards =
            VecDataset::new(dataset.name(), train_data, metric, false).make_shards(max_cardinality);
        let start = Instant::now();
        let cakes = Cakes::new_randomly_sharded(shards, seed, &PartitionCriteria::default());
        let elapsed = start.elapsed().as_secs_f32();
        mt_log!(
            Level::Info,
            "Sharded tree building time: {} s",
            format_f32(elapsed)
        );

        (cakes, elapsed)
    } else {
        let data = VecDataset::new(dataset.name(), train_data, metric, false);

        let start = Instant::now();
        let cakes = Cakes::new(data, seed, &PartitionCriteria::default());
        let elapsed = start.elapsed().as_secs_f32();
        mt_log!(Level::Info, "Tree building time: {} s", format_f32(elapsed));

        (cakes, elapsed)
    };

    let shard_sizes = cakes.shard_cardinalities();
    mt_log!(
        Level::Info,
        "Shard sizes: [{}]",
        shard_sizes
            .iter()
            .map(|s| s.to_formatted_string(&num_format::Locale::en))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let tuning_time = if let Some((tuning_depth, tuning_k)) = tuning {
        mt_log!(Level::Info, "Tuning depth: {}", tuning_depth);
        mt_log!(Level::Info, "Tuning k: {}", tuning_k);

        let start = Instant::now();
        cakes.auto_tune_knn(tuning_k, tuning_depth);
        let elapsed = start.elapsed().as_secs_f32();
        mt_log!(Level::Info, "Tuning time: {} s", format_f32(elapsed));

        let algorithm = cakes.tuned_knn_algorithm();
        mt_log!(Level::Info, "Tuned algorithm: {}", algorithm.name());

        Some(elapsed)
    } else {
        None
    };

    for &k in ks {
        mt_log!(Level::Info, "k: {k}");

        let algorithm_throughput = if tuning.is_some() {
            let mut num_queries_ran = 0;
            let mut hits = Vec::new();
            let start = Instant::now();
            for chunk in queries.chunks(100) {
                hits.extend(cakes.batch_tuned_knn_search(chunk, k));
                num_queries_ran += chunk.len();
                if start.elapsed().as_secs_f32() > max_search_time {
                    break;
                }
            }
            let elapsed = start.elapsed().as_secs_f32();
            let throughput = num_queries_ran.as_f32() / elapsed;

            mt_log!(Level::Trace, "Hits: {hits:?}");
            mt_log!(
                Level::Info,
                "Tuned Throughput: {} QPS",
                format_f32(throughput)
            );

            vec![(cakes.tuned_knn_algorithm().name().to_string(), throughput)]
        } else {
            let mut algorithm_throughput = Vec::new();
            for &algo in knn::Algorithm::variants() {
                let mut num_queries_ran = 0;
                let mut hits = Vec::new();
                let start = Instant::now();
                for chunk in queries.chunks(100) {
                    hits.extend(cakes.batch_knn_search(chunk, k, algo));
                    num_queries_ran += chunk.len();
                    if start.elapsed().as_secs_f32() > max_search_time {
                        break;
                    }
                }
                let elapsed = start.elapsed().as_secs_f32();
                let throughput = num_queries_ran.as_f32() / elapsed;

                mt_log!(Level::Trace, "Hits: {hits:?}");
                mt_log!(
                    Level::Info,
                    "Throughput for {}: {} QPS",
                    algo.name(),
                    format_f32(throughput)
                );

                algorithm_throughput.push((algo.name().to_string(), throughput));
            }
            algorithm_throughput
        };

        let mut num_queries_ran = 0;
        let mut linear_hits = Vec::new();
        let start = Instant::now();
        for chunk in queries.chunks(100) {
            linear_hits.extend(cakes.batch_linear_knn_search(chunk, k));
            num_queries_ran += chunk.len();
            if start.elapsed().as_secs_f32() > max_search_time {
                break;
            }
        }
        let linear_elapsed = start.elapsed().as_secs_f32();
        let linear_throughput = num_queries_ran.as_f32() / linear_elapsed;

        mt_log!(Level::Trace, "Linear hits: {linear_hits:?}");

        mt_log!(
            Level::Info,
            "Linear throughput: {} QPS",
            format_f32(linear_throughput)
        );

        Report {
            dataset: dataset.name(),
            metric: dataset.metric_name().to_string(),
            cardinality,
            dimensionality,
            shard_sizes: shard_sizes.clone(),
            num_queries,
            tree_build_time,
            tuning_parameters: tuning,
            tuning_time,
            k,
            algorithm_throughput,
            linear_throughput,
        }
        .save(output_dir)?;
    }

    Ok(())
}

/// A report of the results of an ANN benchmark.
#[derive(Debug, Serialize, Deserialize)]
struct Report {
    /// Name of the data set.
    dataset: String,
    /// Name of the distance function.
    metric: String,
    /// Number of data points in the data set.
    cardinality: usize,
    /// Dimensionality of the data set.
    dimensionality: usize,
    /// Sizes of the shards created for `ShardedCakes`.
    shard_sizes: Vec<usize>,
    /// Number of queries used for search.
    num_queries: usize,
    /// The time taken to build the tree.
    tree_build_time: f32,
    /// The parameters used for auto-tuning.
    tuning_parameters: Option<(usize, usize)>,
    /// The time taken to auto-tune.
    tuning_time: Option<f32>,
    /// Number of nearest neighbors to search for.
    k: usize,
    /// The name and search throughput of knn algorithms.
    algorithm_throughput: Vec<(String, f32)>,
    /// Throughput of linear search.
    linear_throughput: f32,
}

impl Report {
    /// Save the report to a file in the given directory.
    fn save(&self, dir: &Path) -> Result<(), String> {
        let path = dir.join(format!("knn-{}-{}.json", self.dataset, self.k));
        let report = serde_json::to_string_pretty(&self).map_err(|e| e.to_string())?;
        std::fs::write(path, report).map_err(|e| e.to_string())?;
        Ok(())
    }
}
