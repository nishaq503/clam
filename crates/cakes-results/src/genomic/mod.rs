//! Cakes benchmarks on genomic datasets.

pub mod metrics;
mod nucleotides;
mod read_silva;
pub mod sequence;

use core::cmp::Ordering;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use abd_clam::{knn, rnn, Cakes, Dataset, VecDataset};
use clap::Parser;
use distances::Number;
use mt_logger::{mt_level, mt_log, mt_stream, Level, OutputStream};
use sequence::Sequence;
use serde::{Deserialize, Serialize};

use nucleotides::Nucleotide;

/// Runs genomic sequence search.
///
/// # Arguments
///
/// * `input_dir`: The directory containing the input data.
/// * `dataset_name`: The name of the data set to search.
/// * `metric_name`: The name of the metric to use for computing distances.
/// * `seed`: Optional seed for the random number generator.
/// * `tuning_depth`: The depth of the tree to use for auto-tuning knn-search.
/// * `tuning_k`: The value of k to use for auto-tuning knn-search.
/// * `ks`: The values of k to use for knn-search.
/// * `rs`: The values of r to use for range search.
/// * `output_dir`: The directory to save the results to.
#[allow(clippy::too_many_arguments)]
pub fn run(
    input_dir: &Path,
    metric_name: &str,
    sample_size: Option<usize>,
    seed: Option<u64>,
    tuning_depth: usize,
    tuning_k: usize,
    ks: &[usize],
    rs: &[usize],
    output_dir: &Path,
) -> Result<(), String> {
    mt_stream!(OutputStream::Both);
    mt_level!(Level::Debug);

    // Parse the metric.
    let metric = Metric::from_str(metric_name)?;
    let metric_name = metric.name();
    mt_log!(Level::Info, "Using metric: {metric_name}");

    let is_expensive = metric.is_expensive();
    let metric = metric.metric();

    // Check that the data set exists.
    let dataset_name = "silva-SSU-Ref";
    let data_paths = [
        input_dir.join(format!("{dataset_name}-unaligned.txt")),
        input_dir.join(format!("{dataset_name}-alphabet.txt")),
        input_dir.join(format!("{dataset_name}-headers.txt")),
    ];
    for path in &data_paths {
        if !path.exists() {
            return Err(format!("File {path:?} does not exist."));
        }
    }
    let [unaligned_path, _, headers_path] = data_paths;
    mt_log!(Level::Info, "Using data from {unaligned_path:?}.");

    let [train, query_data] = read_silva::silva_to_dataset(
        sample_size,
        &unaligned_path,
        &headers_path,
        metric,
        is_expensive,
    )?;
    mt_log!(
        Level::Info,
        "Read {} data set. Cardinality: {}",
        train.name(),
        train.cardinality()
    );
    mt_log!(
        Level::Info,
        "Read {} query set. Cardinality: {}",
        query_data.name(),
        query_data.cardinality()
    );

    // Check if the cakes data structure already exists.
    let data_size = train.cardinality();
    let cakes_dir = input_dir.join(format!("{dataset_name}-{data_size}-cakes"));

    let (cakes, built, build_time) = if cakes_dir.exists() {
        mt_log!(Level::Info, "Loading search tree from {cakes_dir:?} ...");
        let start = Instant::now();
        let cakes = Cakes::load(&cakes_dir, metric, is_expensive)?;
        let load_time = start.elapsed().as_secs_f32();
        mt_log!(
            Level::Info,
            "Loaded search tree in {load_time:.2e} seconds."
        );

        // let tuned_algorithm = cakes.tuned_knn_algorithm();
        // let tuned_algorithm = tuned_algorithm.name();
        // mt_log!(Level::Info, "Tuned algorithm is {tuned_algorithm}");

        (cakes, false, load_time)
    } else {
        let criteria = abd_clam::PartitionCriteria::new(true)
            .with_min_cardinality(10)
            .with_max_depth(128);

        mt_log!(Level::Info, "Creating search tree ...");
        let start = Instant::now();
        let mut cakes = Cakes::new(train, seed, &criteria);
        let build_time = start.elapsed().as_secs_f32();
        mt_log!(
            Level::Info,
            "Created search tree in {build_time:.2e} seconds."
        );

        mt_log!(
            Level::Info,
            "Tuning knn-search with k {tuning_k} and depth {tuning_depth} ..."
        );

        let start = Instant::now();
        cakes.auto_tune_knn(tuning_depth, tuning_k);
        let tuning_time = start.elapsed().as_secs_f32();
        mt_log!(
            Level::Info,
            "Tuned knn-search in {tuning_time:.2e} seconds."
        );

        let tuned_algorithm = cakes.tuned_knn_algorithm();
        let tuned_algorithm = tuned_algorithm.name();
        mt_log!(Level::Info, "Tuned algorithm is {tuned_algorithm}");

        // Save the Cakes data structure.
        std::fs::create_dir(&cakes_dir).map_err(|e| e.to_string())?;
        cakes.save(&cakes_dir)?;

        (cakes, true, build_time)
    };

    measure_throughput(
        &cakes,
        built,
        build_time,
        ks,
        rs,
        &query_data,
        dataset_name,
        metric_name,
        output_dir,
    )?;

    Ok(())
}

/// Measure the throughput of the tuned algorithm.
///
/// # Arguments
///
/// * `cakes`: The cakes data structure.
/// * `built`: Whether the cakes data structure was built or loaded from disk.
/// * `build_time`: The time taken to build the cakes data structure or load it
/// from disk.
/// * `args`: The command line arguments.
/// * `queries`: The queries.
/// * `query_headers`: The headers of the queries.
/// * `train_headers`: The headers of the training set.
/// * `stem`: The stem of the data set name.
/// * `metric_name`: The name of the metric.
/// * `output_dir`: The output directory.
///
/// # Errors
///
/// * If the `output_dir` does not exist or cannot be written to.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_possible_truncation
)]
fn measure_throughput(
    cakes: &Cakes<Sequence, u32, VecDataset<Sequence, u32, String>>,
    built: bool,
    build_time: f32,
    ks: &[usize],
    rs: &[usize],
    query_data: &VecDataset<Sequence, u32, String>,
    dataset_name: &str,
    metric_name: &str,
    output_dir: &Path,
) -> Result<(), String> {
    let train = cakes.shards()[0];
    let queries = query_data.data().iter().collect::<Vec<_>>();
    let run_linear = train.cardinality() < 10_000;

    // Run the linear search algorithm.
    let k = ks
        .iter()
        .max()
        .copied()
        .unwrap_or_else(|| unreachable!("No ks!"));

    let (linear_results, linear_throughput) = if run_linear {
        let num_linear_queries = 10;
        let start = Instant::now();
        let linear_results = cakes.batch_linear_knn_search(&queries[..num_linear_queries], k);
        let linear_search_time = start.elapsed().as_secs_f32();
        let linear_throughput = queries.len().as_f32() / linear_search_time;
        let linear_throughput = linear_throughput / num_linear_queries.as_f32();
        mt_log!(
            Level::Info,
            "With k = {k}, achieved linear search throughput of {linear_throughput:.2e} QPS."
        );
        (linear_results, linear_throughput)
    } else {
        mt_log!(
            Level::Info,
            "Not running linear search because the training set is too large."
        );
        (vec![], -1.0)
    };

    // Perform knn-search for each value of k on all queries.
    for &k in ks {
        mt_log!(Level::Info, "Starting knn-search with k = {k} ...");

        // Run each algorithm.
        for &algo in knn::Algorithm::variants() {
            mt_log!(Level::Info, "Running algorithm {} ...", algo.name());

            let start = Instant::now();
            let results = cakes.batch_knn_search(&queries, k, algo);
            let search_time = start.elapsed().as_secs_f32();
            let throughput = queries.len().as_f32() / search_time;
            mt_log!(
                Level::Info,
                "With k = {k}, {} achieved throughput of {throughput:.2e} QPS.",
                algo.name()
            );

            let (mean_recall, _) = if run_linear {
                let (mean_recall, hits) =
                    recall_and_hits(results.clone(), &linear_results, train, query_data);
                mt_log!(
                    Level::Debug,
                    "With k = {k}, achieved mean recall of {mean_recall:.3}."
                );
                (mean_recall, hits)
            } else {
                mt_log!(
                    Level::Debug,
                    "Not computing recall because the training set is too large."
                );
                (0.0, vec![])
            };

            // Create the report.
            let report = Report {
                dataset: dataset_name,
                metric: metric_name,
                cardinality: train.cardinality(),
                built,
                build_time,
                shard_sizes: cakes.shard_cardinalities(),
                num_queries: queries.len(),
                kind: "knn",
                val: k,
                algorithm: algo.name(),
                throughput,
                linear_throughput,
                mean_recall,
                // hits,
            };

            // Save the report.
            report.save(output_dir)?;
        }
    }

    // Run the linear search algorithm.
    let radius = rs
        .iter()
        .max()
        .copied()
        .unwrap_or_else(|| unreachable!("No radii!")) as u32
        * Nucleotide::gap_penalty();

    let (linear_results, linear_throughput) = if run_linear {
        let num_linear_queries = 10;
        let start = Instant::now();
        let linear_results = cakes.batch_linear_rnn_search(&queries[..num_linear_queries], radius);
        let linear_search_time = start.elapsed().as_secs_f32();
        let linear_throughput = queries.len().as_f32() / linear_search_time;
        let linear_throughput = linear_throughput / num_linear_queries.as_f32();
        mt_log!(
            Level::Info,
            "With r = {radius}, achieved linear search throughput of {linear_throughput:.2e} QPS."
        );
        (linear_results, linear_throughput)
    } else {
        mt_log!(
            Level::Info,
            "Not running linear search because the training set is too large."
        );
        (vec![], -1.0)
    };

    // Perform range search for each value of r on all queries.
    for r in rs.iter().map(|&r| r as u32 * Nucleotide::gap_penalty()) {
        mt_log!(Level::Info, "Starting range search with r = {r} ...");

        // Run the tuned algorithm.
        let start = Instant::now();
        let results = cakes.batch_rnn_search(&queries, radius, rnn::Algorithm::Clustered);
        let search_time = start.elapsed().as_secs_f32();
        let throughput = queries.len().as_f32() / search_time;
        mt_log!(
            Level::Info,
            "With r = {r}, {} achieved throughput of {throughput:.2e} QPS.",
            rnn::Algorithm::Clustered.name()
        );

        let (mean_recall, _) = if run_linear {
            let (mean_recall, hits) =
                recall_and_hits(results.clone(), &linear_results, train, query_data);
            mt_log!(
                Level::Debug,
                "With r = {r}, achieved mean recall of {mean_recall:.3}."
            );
            (mean_recall, hits)
        } else {
            mt_log!(
                Level::Debug,
                "Not computing recall because the training set is too large."
            );
            (0.0, vec![])
        };

        // Create the report.
        let report = Report {
            dataset: dataset_name,
            metric: metric_name,
            cardinality: train.cardinality(),
            built,
            build_time,
            shard_sizes: cakes.shard_cardinalities(),
            num_queries: queries.len(),
            kind: "rnn",
            val: r as usize,
            algorithm: rnn::Algorithm::Clustered.name(),
            throughput,
            linear_throughput,
            mean_recall,
            // hits,
        };

        // Save the report.
        report.save(output_dir)?;
    }

    Ok(())
}

/// Compute the recall and hits of the tuned algorithm.
///
/// # Arguments
///
/// * `results`: The results of the tuned algorithm.
/// * `linear_results`: The results of linear search.
/// * `queries`: The queries.
/// * `query_headers`: The headers of the queries.
/// * `train_headers`: The headers of the training set.
/// * `train`: The training set.
///
/// # Returns
///
/// * The mean recall of the tuned algorithm.
/// * The headers of hits of the tuned algorithm.
#[allow(clippy::type_complexity)]
fn recall_and_hits(
    results: Vec<Vec<(usize, u32)>>,
    linear_results: &[Vec<(usize, u32)>],
    train: &VecDataset<Sequence, u32, String>,
    query_data: &VecDataset<Sequence, u32, String>,
) -> (f32, Vec<(String, Vec<(String, u32)>)>) {
    // Compute the recall of the tuned algorithm.
    let mean_recall = results
        .iter()
        .zip(linear_results)
        .map(|(hits, linear_hits)| compute_recall(hits.clone(), linear_hits.clone()))
        .sum::<f32>()
        / linear_results.len().as_f32();

    let train_headers = train.metadata();
    let query_headers = query_data.metadata();
    let hits = results
        .into_iter()
        .zip(query_headers.iter())
        .map(|(hits, qh)| {
            let hit_headers = hits
                .iter()
                .map(|&(i, d)| (train_headers[i].clone(), d))
                .collect::<Vec<_>>();
            (qh.clone(), hit_headers)
        })
        .collect::<Vec<_>>();

    (mean_recall, hits)
}

/// CLI arguments for the genomic benchmarks.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the directory containing the input data. This directory should
    /// contain the silva-18s unaligned sequences file, along with the files
    /// containing the headers and alphabet information.
    #[arg(long)]
    input_dir: PathBuf,
    /// Output directory for the report.
    #[arg(long)]
    output_dir: PathBuf,
    /// The metric to use for computing distances. One of "hamming",
    /// "levenshtein", or "needleman-wunsch".
    #[arg(long)]
    metric: String,
    /// The depth of the tree to use for auto-tuning knn-search.
    #[arg(long, default_value = "7")]
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
    /// Seed for the random number generator.
    #[arg(long)]
    seed: Option<u64>,
}

/// Metrics for computing distances between genomic sequences.
#[derive(Debug)]
enum Metric {
    /// Hamming distance.
    Hamming,
    /// Levenshtein distance.
    Levenshtein,
    /// Needleman-Wunsch distance.
    NeedlemanWunsch,
}

impl Metric {
    /// Return the name of the metric.
    const fn name(&self) -> &str {
        match self {
            Self::Hamming => "hamming",
            Self::Levenshtein => "levenshtein",
            Self::NeedlemanWunsch => "needleman-wunsch",
        }
    }

    /// Return the metric corresponding to the given name.
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "hamming" | "ham" => Ok(Self::Hamming),
            "levenshtein" | "lev" => Ok(Self::Levenshtein),
            "needleman-wunsch" | "needlemanwunsch" | "nw" => Ok(Self::NeedlemanWunsch),
            _ => Err(format!("Unknown metric: {s}")),
        }
    }

    /// Return the metric function.
    #[allow(clippy::ptr_arg)]
    fn metric(&self) -> fn(&Sequence, &Sequence) -> u32 {
        match self {
            Self::Hamming => metrics::hamming,
            Self::Levenshtein => metrics::levenshtein,
            Self::NeedlemanWunsch => metrics::needleman_wunsch,
        }
    }

    /// Return whether the metric is expensive to compute.
    ///
    /// The Hamming distance is cheap to compute, while the Levenshtein and
    /// Needleman-Wunsch distances are expensive to compute.
    const fn is_expensive(&self) -> bool {
        !matches!(self, Self::Hamming)
    }
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
    /// The kind of search performed. One of "knn" or "rnn".
    kind: &'a str,
    /// Value of k used for knn-search or value of r used for range search.
    val: usize,
    /// Name of the algorithm used.
    algorithm: &'a str,
    /// Throughput of the tuned algorithm.
    throughput: f32,
    /// Throughput of linear search.
    linear_throughput: f32,
    /// Mean recall of the tuned algorithm.
    mean_recall: f32,
    // /// Hits for each query.
    // hits: Vec<(String, Vec<(String, u32)>)>,
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
            "{}-{}-{}-{}-{}.json",
            self.dataset, self.cardinality, self.kind, self.algorithm, self.val
        ));
        let report = serde_json::to_string_pretty(&self).map_err(|e| e.to_string())?;
        std::fs::write(path, report).map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Compute the recall of a knn-search algorithm.
///
/// # Arguments
///
/// * `hits`: the hits of the algorithm.
/// * `linear_hits`: the hits of linear search.
///
/// # Returns
///
/// * The recall of the algorithm.
#[must_use]
pub fn compute_recall<U: Number>(
    mut hits: Vec<(usize, U)>,
    mut linear_hits: Vec<(usize, U)>,
) -> f32 {
    if linear_hits.is_empty() {
        if hits.is_empty() {
            1.0
        } else {
            0.0
        }
    } else if hits.is_empty() {
        0.0
    } else {
        let num_hits = hits.len();

        hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
        let mut hits = hits.into_iter().map(|(_, d)| d).peekable();

        linear_hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
        let mut linear_hits = linear_hits.into_iter().map(|(_, d)| d).peekable();

        let mut num_common = 0_usize;
        while let (Some(&hit), Some(&linear_hit)) = (hits.peek(), linear_hits.peek()) {
            if (hit - linear_hit).abs() <= U::epsilon() {
                num_common += 1;
                hits.next();
                linear_hits.next();
            } else if hit < linear_hit {
                hits.next();
            } else {
                linear_hits.next();
            }
        }

        // TODO: divide by the number of linear hits?
        num_common.as_f32() / num_hits.as_f32()
    }
}
