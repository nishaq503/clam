//! Builds and writes the MSA tree for MuSALS

use std::path::Path;

use crate::{commands::musals::ShellCostMatrix, data::ShellData, metrics::Metric, trees::ShellTree};

/// Builds and writes the MSA tree for MuSALS.
pub fn build_msa<P: AsRef<Path>>(
    inp_data: ShellData,
    metric: &Metric,
    cost_matrix: &ShellCostMatrix,
    out_dir: P,
    save_fasta: bool,
) -> Result<(), String> {
    let out_dir = out_dir.as_ref();
    let parent = out_dir
        .parent()
        .ok_or_else(|| format!("Output path '{out_dir:?}' has no parent directory"))?;
    if !parent.exists() {
        return Err(format!("Output directory '{parent:?}' does not exist"));
    }
    if !out_dir.exists() {
        std::fs::create_dir_all(out_dir).map_err(|e| format!("Failed to create output directory '{out_dir:?}': {e}"))?;
    }

    let suffix = format!("msa-{cost_matrix}");
    let cost_matrix = cost_matrix.get();

    let tree = ShellTree::new(inp_data, metric)?;
    let msa_tree = match tree {
        ShellTree::Levenshtein(tree) => ShellTree::Levenshtein(tree.par_into_msa(&cost_matrix)),
        _ => return Err("MSA tree can only be built for string data.".to_string()),
    };
    println!("Writing MSA tree to '{out_dir:?}' with suffix '{suffix}'...");

    msa_tree.write_to(out_dir, Some(&suffix))?;

    if save_fasta {
        let items = match msa_tree {
            ShellTree::Levenshtein(tree) => tree.take_items(),
            _ => unreachable!(),
        };
        let data = ShellData::String(items);

        let fasta_path = out_dir.join(format!("{suffix}.fasta"));
        println!("Saving MSA sequences to '{fasta_path:?}'...");

        data.write(fasta_path)?;
    }

    Ok(())
}
