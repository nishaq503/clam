//! Create the Silva-18S dataset for use in CLAM.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use abd_clam::{Dataset, VecDataset};
use distances::Number;
use mt_logger::{mt_log, Level};
use rand::prelude::*;
use rayon::prelude::*;

use super::sequence::Sequence;

/// Read the Silva-18S dataset from the given path.
///
/// The `unaligned_path` should point to the `silva-SSU-Ref-unaligned.txt` file
/// which contains the unaligned sequences, one per line.
///
/// The `metric` function should compute the distance between two sequences.
///
/// The dataset is (randomly) split into a query set of `1_000` sequences and a
/// training set of the remaining sequences.
///
/// # Arguments
///
/// * `unaligned_path`: The path to the unaligned sequences file.
/// * `headers_path`: The path to the headers file.
/// * `metric`: The metric to use for computing distances.
/// * `is_expensive`: Whether the metric is expensive to compute.
///
/// # Returns
///
/// 4 datasets: the training set, the query set, the training set headers, and
/// the query set headers.
///
/// # Errors
///
/// * If the file at `unaligned_path` does not exist, cannot be read, is not
/// valid UTF-8, or is otherwise malformed.
#[allow(clippy::ptr_arg)]
pub fn silva_to_dataset(
    sample_size: Option<usize>,
    unaligned_path: &Path,
    headers_path: &Path,
    metric: fn(&Sequence, &Sequence) -> u32,
    is_expensive: bool,
) -> Result<[VecDataset<Sequence, u32, String>; 2], String> {
    // Get the stem of the file name.
    let stem = unaligned_path
        .file_stem()
        .ok_or_else(|| format!("Could not get file stem for {unaligned_path:?}"))?;
    let stem = stem
        .to_str()
        .ok_or_else(|| format!("Could not convert file stem to string for {unaligned_path:?}"))?;

    // Open the unaligned sequences file and read the lines.
    let file = File::open(unaligned_path)
        .map_err(|e| format!("Could not open file {unaligned_path:?}: {e}"))?;
    let reader = BufReader::new(file);
    let sequences = reader
        .lines()
        .map(|line| line.map_err(|e| format!("Could not read line: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    mt_log!(
        Level::Info,
        "Read {} sequences from {unaligned_path:?}.",
        sequences.len()
    );

    // Compute and log some statistics about the sequences.
    let seq_lengths = sequences.iter().map(String::len).collect::<Vec<_>>();
    let min_len = seq_lengths
        .iter()
        .min()
        .unwrap_or_else(|| unreachable!("No sequences!"));
    let max_len = seq_lengths
        .iter()
        .max()
        .unwrap_or_else(|| unreachable!("No sequences!"));
    let (mean, std_dev) = {
        let sum = seq_lengths.iter().sum::<usize>().as_f64();
        let mean = sum / seq_lengths.len().as_f64();
        let variance = seq_lengths
            .iter()
            .map(|&len| (len.as_f64() - mean).powi(2))
            .sum::<f64>()
            / seq_lengths.len().as_f64();
        let std_dev = variance.sqrt();
        (mean, std_dev)
    };
    mt_log!(Level::Info, "Minimum sequence length: {min_len}");
    mt_log!(Level::Info, "Maximum sequence length: {max_len}");
    mt_log!(Level::Info, "Mean sequence length: {mean:.3}");
    mt_log!(
        Level::Info,
        "Standard deviation of sequence length: {std_dev:.3}"
    );

    // Read the headers file.
    let file = File::open(headers_path)
        .map_err(|e| format!("Could not open file {headers_path:?}: {e}"))?;
    let reader = BufReader::new(file);
    let headers = reader
        .lines()
        .map(|line| line.map_err(|e| format!("Could not read line: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    mt_log!(
        Level::Info,
        "Read {} headers from {headers_path:?}.",
        headers.len()
    );

    // Convert the lines into sequences.
    let sequences = sequences
        .par_iter()
        .map(|s| Sequence::from_str(s))
        .collect::<Result<Vec<_>, _>>()?;

    let (train, queries) = {
        // join the lines and headers into a single vector of (line, header) pairs.
        let mut sequences = sequences.into_iter().zip(headers).collect::<Vec<_>>();

        // If a sample size was specified, take a random sample of that size.
        let num_queries = sample_size.map_or(100, |sample_size| {
            let num_queries = (sample_size / 10).min(100);

            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            sequences.shuffle(&mut rng);
            sequences.truncate(sample_size + num_queries);

            num_queries
        });

        (sequences.split_off(num_queries), sequences)
    };

    // Create the training dataset and use corresponding headers as metadata.
    let (train, train_headers): (Vec<_>, Vec<_>) = train.into_iter().unzip();
    let train = VecDataset::new(format!("{stem}-train"), train, metric, is_expensive)
        .assign_metadata(train_headers)?;
    mt_log!(
        Level::Info,
        "Using {} sequences for training.",
        train.cardinality()
    );

    // Create the query dataset and use corresponding headers as metadata.
    let (queries, query_headers): (Vec<_>, Vec<_>) = queries.into_iter().unzip();
    let queries = VecDataset::new(format!("{stem}-queries"), queries, metric, is_expensive)
        .assign_metadata(query_headers)?;
    mt_log!(
        Level::Info,
        "Using {} sequences for queries.",
        queries.cardinality()
    );

    Ok([train, queries])
}
