//! Compression and compressive search with Pancakes

use clap::Subcommand;

/// Compress a pre-built tree, and search compressed trees.
#[derive(Subcommand, Debug, serde::Serialize, serde::Deserialize)]
pub enum Action {
    /// Compress a pre-built tree.
    Compress {
        /// The name of the `codec` to use for compression.
        #[clap(short('c'), long)]
        codec: String,
    },
}
