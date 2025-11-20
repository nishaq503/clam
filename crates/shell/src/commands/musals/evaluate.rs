//! Evaluate MSA quality metrics.

use std::path::Path;

use crate::{data::OutputFormat, trees::ShellTree};

/// Evaluate MSA quality metrics.
pub fn evaluate_msa<P: AsRef<Path>>(
    tree_dir: P,
    quality_metrics: &[super::ShellQualityMetric],
    cost_matrix: &super::ShellCostMatrix,
    out_path: P,
) -> Result<(), String> {
    let out_path = out_path.as_ref();

    println!("Output path: {out_path:?}");
    let out_dir = out_path
        .parent()
        .ok_or_else(|| "Output path must have a parent directory.".to_string())?;
    if !out_dir.exists() {
        std::fs::create_dir_all(out_dir).map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    let suffix = out_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "Output path must have a file name.".to_string())?;

    let ext = out_path
        .extension()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "Output path must have a file extension.".to_string())?;

    let (tree, tree_path) = ShellTree::read_from(tree_dir)?;
    println!("Read tree from {tree_path:?}.");

    let tree = match tree {
        ShellTree::Levenshtein(tree) => tree,
        _ => return Err("MUSALS evaluation currently only supports Levenshtein trees.".to_string()),
    };
    let cost_matrix = cost_matrix.get();

    let results = quality_metrics
        .iter()
        .map(|m| m.get())
        .map(|m| tree.par_compute_quality_metric(&m, &cost_matrix));

    for m in results {
        let file_name = format!("{}-{suffix}.{ext}", m.name());
        let out_path = out_dir.join(file_name);
        OutputFormat::write(&out_path, &m)?;
    }

    Ok(())
}
