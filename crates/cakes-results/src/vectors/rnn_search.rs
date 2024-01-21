//! Benchmark the RNN search algorithm.

use std::{path::Path, time::Instant};

use abd_clam::{rnn, Cakes, Dataset, PartitionCriteria, VecDataset};
use distances::Number;
use mt_logger::{mt_log, Level};
use num_format::ToFormattedString;
use serde::{Deserialize, Serialize};

use crate::{
    utils::{compute_recall, format_f32},
    vectors::ann_datasets::AnnDatasets,
};

/// Report the results of an ANN benchmark.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn rnn_search(
    input_dir: &Path,
    dataset: &str,
    use_shards: bool,
    max_search_time: f32,
    radius_divisors: &[u32],
    seed: Option<u64>,
    output_dir: &Path,
) -> Result<(), String> {
    mt_log!(Level::Info, "Start rnn_search on {dataset}");

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

    let cakes = if use_shards {
        let max_cardinality = if cardinality < 1_000_000 {
            cardinality
        } else if cardinality < 5_000_000 {
            100_000
        } else {
            1_000_000
        };

        let shards =
            VecDataset::new(dataset.name(), train_data, metric, false).make_shards(max_cardinality);
        Cakes::new_randomly_sharded(shards, seed, &PartitionCriteria::default())
    } else {
        let data = VecDataset::new(dataset.name(), train_data, metric, false);
        Cakes::new(data, seed, &PartitionCriteria::default())
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

    let root_radius = cakes.trees()[0].root().radius();

    for &rd in radius_divisors {
        let radius = root_radius / rd.as_f32();
        mt_log!(Level::Info, "radius divisor: {rd}, radius: {radius}");

        // Run Clustered RNN search
        let mut num_queries_ran = 0;
        let mut hits = Vec::new();
        let start = Instant::now();
        for chunk in queries.chunks(100) {
            let chunk_hits = cakes.batch_rnn_search(chunk, radius, rnn::Algorithm::Clustered);
            num_queries_ran += chunk.len();
            hits.extend(chunk_hits);
            if start.elapsed().as_secs_f32() > max_search_time {
                break;
            }
        }
        let elapsed = start.elapsed().as_secs_f32();
        let throughput = num_queries_ran.as_f32() / elapsed;
        mt_log!(Level::Info, "Throughput: {} QPS", format_f32(throughput));

        // Run linear search
        num_queries_ran = 0;
        let mut linear_hits = Vec::new();
        let start = Instant::now();
        for chunk in queries.chunks(100) {
            let chunk_hits = cakes.batch_linear_rnn_search(chunk, radius);
            num_queries_ran += chunk.len();
            linear_hits.extend(chunk_hits);
            if start.elapsed().as_secs_f32() > max_search_time {
                break;
            }
        }
        let linear_elapsed = start.elapsed().as_secs_f32();
        let linear_throughput = num_queries_ran.as_f32() / linear_elapsed;
        mt_log!(
            Level::Info,
            "Linear throughput: {} QPS",
            format_f32(linear_throughput)
        );

        let speedup_factor = throughput / linear_throughput;
        mt_log!(
            Level::Info,
            "Speedup factor: {}",
            format_f32(speedup_factor)
        );

        let recall = hits
            .into_iter()
            .zip(linear_hits)
            .map(|(hits, linear_hits)| compute_recall(hits, linear_hits))
            .sum::<f32>()
            / num_queries_ran.as_f32();
        mt_log!(Level::Info, "Recall: {}", format_f32(recall));

        Report {
            dataset: &dataset.name(),
            metric: dataset.metric_name(),
            cardinality,
            dimensionality,
            shard_sizes: shard_sizes.clone(),
            num_queries,
            radius,
            throughput,
            linear_throughput,
        }
        .save(output_dir, rd)?;
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
    /// Dimensionality of the data set.
    dimensionality: usize,
    /// Sizes of the shards created for `ShardedCakes`.
    shard_sizes: Vec<usize>,
    /// Number of queries used for search.
    num_queries: usize,
    /// Radius used for search.
    radius: f32,
    /// Throughput of the tuned algorithm.
    throughput: f32,
    /// Throughput of linear search.
    linear_throughput: f32,
}

impl Report<'_> {
    /// Save the report to a file in the given directory.
    fn save(&self, dir: &Path, rd: u32) -> Result<(), String> {
        let path = dir.join(format!("rnn-{}-{rd}.json", self.dataset));
        let report = serde_json::to_string_pretty(&self).map_err(|e| e.to_string())?;
        std::fs::write(path, report).map_err(|e| e.to_string())?;
        Ok(())
    }
}
