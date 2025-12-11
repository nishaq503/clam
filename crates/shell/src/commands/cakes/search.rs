//! Search a given tree for some given queries.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{data::ShellData, search::ShellCakes, trees::ShellTree};

/// Represents the complete search results for all queries and algorithms.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResults {
    /// Results for each query
    pub results: Vec<QueryResult>,
}

/// Represents the result for a single query across all algorithms.
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResult {
    /// Index of the query in the instances dataset
    pub query_index: usize,
    /// Results from each algorithm
    pub algorithms: Vec<AlgorithmResult>,
}

/// Represents the result from a single algorithm for a query.
#[derive(Debug, Serialize, Deserialize)]
pub struct AlgorithmResult {
    /// Name of the algorithm used
    pub algorithm: String,
    /// List of (index, distance) pairs for the neighbors found
    pub neighbors: Vec<(usize, f64)>,
}

/// Searches a given tree for some given queries and saves results to a file.
///
/// This function is responsible for searching a given tree for some given queries.
/// For each instance, it applies all query algorithms to the tree and saves the results
/// to a single file in the format determined by the file extension.
///
/// # Arguments
/// * `tree_dir` - The directory containing the tree to search.
/// * `queries` - The dataset containing the query instances.
/// * `algorithms` - A slice of `ShellCakes` algorithms to use for searching.
/// * `out_path` - The path to the output file where results will be saved.
///
/// # Returns
/// A `Result` containing either `Ok(())` if the search was successful, or `Err(String)` if an error occurred.
pub fn search_tree<P: AsRef<Path> + core::fmt::Debug>(tree_dir: P, queries: ShellData, algorithms: &[ShellCakes], out_path: P) -> Result<(), String> {
    // Create parent directory if it doesn't exist
    if let Some(parent) = out_path.as_ref().parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Failed to create parent directory: {e}"))?;
    }

    let (tree, tree_path) = ShellTree::read_from(tree_dir)?;
    println!("Read tree from {tree_path:?}.");

    tree.search(queries, algorithms, out_path)
}
