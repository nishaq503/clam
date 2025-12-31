//! Builds and writes the MSA tree for MuSALS

use std::path::Path;

use crate::{commands::musals::ShellCostMatrix, data::ShellData, metrics::Metric, trees::ShellTree};

/// Builds and writes the MSA tree for MuSALS.
pub fn build_msa<P: AsRef<Path>>(
    inp_data: ShellData,
    metric: &Metric,
    max_recursion_depth: Option<usize>,
    cost_matrix: &ShellCostMatrix,
    out_dir: P,
    save_fasta: bool,
) -> Result<(), String> {
    let out_dir = out_dir.as_ref();
    let parent = out_dir.parent().ok_or_else(|| format!("Output path '{out_dir:?}' has no parent directory"))?;
    if !parent.exists() {
        return Err(format!("Output directory '{parent:?}' does not exist"));
    }
    if !out_dir.exists() {
        std::fs::create_dir_all(out_dir).map_err(|e| format!("Failed to create output directory '{out_dir:?}': {e}"))?;
    }

    let tree = ShellTree::new(inp_data, metric, max_recursion_depth)?;
    ftlog::info!("Writing unaligned tree to '{out_dir:?}'...");
    let tree_path = tree.tree_file_path(out_dir, Some("unaligned"));
    tree.write_to(&tree_path)?;

    let msa_tree = match tree {
        ShellTree::Levenshtein(tree) => {
            let cost_matrix = cost_matrix.get();
            if max_recursion_depth.is_some() {
                ShellTree::Levenshtein(tree.par_into_msa(&cost_matrix))
            } else {
                ShellTree::Levenshtein(tree.par_into_msa_iterative(&cost_matrix))
            }
        }
        _ => return Err("MSA tree can only be built for string data.".to_string()),
    };

    let suffix = format!("msa-{cost_matrix}");
    ftlog::info!("Writing MSA tree to '{out_dir:?}' with suffix '{suffix}'...");
    let msa_tree_path = msa_tree.tree_file_path(out_dir, Some(&suffix));
    msa_tree.write_to(&msa_tree_path)?;

    if save_fasta {
        let items = match msa_tree {
            ShellTree::Levenshtein(tree) => tree.take_items(),
            _ => unreachable!(),
        };
        let data = ShellData::String(items);

        let fasta_path = out_dir.join(format!("{suffix}.fasta"));
        ftlog::info!("Saving MSA sequences to '{fasta_path:?}'...");

        data.write(fasta_path)?;
    }

    Ok(())
}
