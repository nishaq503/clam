//! Evaluate MSA quality metrics.

use std::path::Path;

use abd_clam::{DistanceValue, Tree, musals::Sequence};

use crate::{data::OutputFormat, metrics::Metric, trees::ShellTree};

/// Evaluate MSA quality metrics.
pub fn evaluate_msa<P: AsRef<Path>>(
    tree_dir: P,
    metric: &Metric,
    quality_metrics: &[super::ShellQualityMetric],
    sample_size: Option<usize>,
    out_path: P,
) -> Result<(), String> {
    let out_path = out_path.as_ref();

    ftlog::info!("Output path: {out_path:?}");
    let out_dir = out_path.parent().ok_or_else(|| "Output path must have a parent directory.".to_string())?;
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

    // Find a ".bin" file whose name starts with "tree-".
    let tree_dir = tree_dir.as_ref();
    let tree_path = std::fs::read_dir(tree_dir)
        .map_err(|e| format!("Failed to read input directory {}: {}", tree_dir.display(), e))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("msa-") && name.ends_with(".bin"))
        })
        .ok_or_else(|| format!("No tree file found in directory {}", tree_dir.display()))?;

    ftlog::info!("Reading tree from {tree_path:?}...");
    let tree = ShellTree::read_from(&tree_path, metric)?;
    ftlog::info!("Read tree from {tree_path:?}.");

    match tree {
        ShellTree::Lcs(tree) => {
            eval_msa(&tree, quality_metrics, sample_size, out_dir, suffix, ext)?;
        }
        ShellTree::Levenshtein(tree) => {
            eval_msa(&tree, quality_metrics, sample_size, out_dir, suffix, ext)?;
        }
        _ => return Err("MUSALS evaluation currently only supports Levenshtein and Lcs metrics.".to_string()),
    };

    Ok(())
}

fn eval_msa<Id, S, T, A, M>(
    tree: &Tree<Id, S, T, A, M>,
    quality_metrics: &[super::ShellQualityMetric],
    sample_size: Option<usize>,
    out_dir: &Path,
    suffix: &str,
    ext: &str,
) -> Result<(), String>
where
    Id: Send + Sync,
    S: Sequence + Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&S, &S) -> T + Send + Sync,
{
    for m in quality_metrics {
        ftlog::info!("Computing quality metric: {}", m.name());
        let result = tree.par_compute_quality_metric(&m.get(), tree.metric(), sample_size);
        ftlog::info!(
            "Computed quality metric: {}, mean: {}, std_dev: {}, min: {}, max: {}",
            result.name(),
            result.mean(),
            result.std_dev(),
            result.min(),
            result.max()
        );
        let file_name = format!("{}-{suffix}.{ext}", result.name());
        let out_path = out_dir.join(file_name);
        OutputFormat::write(&out_path, &result)?;
    }

    Ok(())
}
