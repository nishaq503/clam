//! Subcommands for CAKES

mod build;
mod search;

use std::path::PathBuf;

use clap::Subcommand;

pub use build::build_new_tree;
pub use search::{AlgorithmResult, QueryResult, SearchOutputFormat, SearchResults, search_tree};

use crate::search::ShellSearchAlgorithm;

#[derive(Subcommand, Debug)]
pub enum CakesAction {
    Build {
        /// The path to the output directory. We will construct the output paths
        /// from this directory as we need them.
        #[arg(short('o'), long)]
        out_dir: PathBuf,
    },
    Search {
        /// The path to the tree file.
        #[arg(short('t'), long)]
        tree_path: PathBuf,

        #[arg(short('I'), long)]
        instances_path: PathBuf,

        #[arg(short('q'), long, value_parser = clap::value_parser!(ShellSearchAlgorithm))]
        query_algorithms: Vec<ShellSearchAlgorithm>,

        /// The path to the output file for search results (format determined by extension: .json, .yaml, or .yml).
        #[arg(short('o'), long)]
        output_path: PathBuf,
    },
}
