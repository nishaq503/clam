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

/// Supported output formats based on file extension.
#[derive(Debug, Clone, Copy)]
pub enum SearchOutputFormat {
    Json,
    Yaml,
}

impl SearchOutputFormat {
    /// Determine format from file extension.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => Ok(SearchOutputFormat::Json),
            Some("yaml") | Some("yml") => Ok(SearchOutputFormat::Yaml),
            Some(ext) => Err(format!(
                "Unsupported file extension: '.{ext}'. Supported formats: .json, .yaml, .yml",
            )),
            None => Err("No file extension found. Please specify .json, .yaml, or .yml".to_string()),
        }
    }
}

/// Searches a given tree for some given queries and saves results to a file.
///
/// This function is responsible for searching a given tree for some given queries.
/// For each instance, it applies all query algorithms to the tree and saves the results
/// to a single file in the format determined by the file extension.
///
/// # Arguments
/// * `inp_data` - The input data to search.
/// * `tree_path` - The path to the tree to search.
/// * `instances` - The instances to search for.
/// * `query_algorithms` - The query algorithms to apply.
/// * `output_path` - The path to the output file (format determined by extension).
///
/// # Returns
/// A `Result` containing either `Ok(())` if the search was successful, or `Err(String)` if an error occurred.
pub fn search_tree<P: AsRef<Path>>(
    tree_path: P,
    queries: ShellData,
    algorithms: &[ShellCakes],
    output_path: P,
) -> Result<(), String> {
    // Determine output format from file extension
    let format = SearchOutputFormat::from_path(&output_path)?;

    // Create parent directory if it doesn't exist
    if let Some(parent) = output_path.as_ref().parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Failed to create parent directory: {e}"))?;
    }

    ShellTree::read_from(tree_path)?.search(queries, algorithms, output_path, format)
}
