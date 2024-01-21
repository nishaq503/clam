//! Module for generating csv files from cluster trees.

use std::path::Path;

use abd_clam::PartitionCriteria;
use abd_clam::{Cakes, Instance, VecDataset};
use distances::Number;

use super::genomic;
use super::radio;
use super::vectors::ann_datasets;

/// Save the properties of all clusters in a single tree to a CSV file.
///
/// # Arguments
///
/// * `dataset` - The name of the dataset.
/// * `tree_dir` - The directory containing the tree.
/// * `output_dir` - The directory to save the CSV file to.
///
/// # Errors
///
/// * If the dataset is not valid.
/// * If the tree directory does not exist.
/// * If the tree directory cannot be read.
pub fn save_csv(dataset: &str, tree_dir: &Path, output_dir: &Path) -> Result<(), String> {
    let data_variant = Datasets::from_str(dataset)?;

    match data_variant {
        Datasets::Vectors => {
            // We did not save these trees so we need to rebuild them.

            // Get the dataset and metric.
            let data = ann_datasets::AnnDatasets::from_str(dataset)?;
            let metric = data.metric()?;
            let [data, _] = data.read(tree_dir)?;
            let data = VecDataset::new(dataset.to_string(), data, metric, false);

            // Create the Cakes object.
            let criteria = PartitionCriteria::default();
            let cakes = Cakes::new(data, None, &criteria);

            let tree_dir = {
                let mut tree_dir = tree_dir.to_path_buf();
                tree_dir.pop();
                tree_dir.push("trees");
                tree_dir.push(dataset);
                tree_dir
            };

            // If the tree directory exists, delete it. Otherwise, create it.
            if tree_dir.exists() {
                std::fs::remove_dir_all(&tree_dir)
                    .map_err(|e| format!("Could not delete tree directory: {e}"))?;
            }
            std::fs::create_dir(&tree_dir)
                .map_err(|e| format!("Could not create tree directory: {e}"))?;

            // Save the tree.
            cakes.save(&tree_dir)?;

            cakes_to_csv::<_, _, bool>(&tree_dir, metric, false, output_dir, dataset)
        }
        Datasets::Genomic => {
            let metric = genomic::metrics::levenshtein;
            cakes_to_csv::<_, _, String>(tree_dir, metric, true, output_dir, dataset)
        }
        Datasets::RadioML => {
            let metric = radio::metrics::dtw;
            cakes_to_csv::<_, _, radio::RadioMLMetadata>(
                tree_dir, metric, true, output_dir, dataset,
            )
        }
    }
}

/// The dataset variants for the Cakes paper.
enum Datasets {
    /// The AnnDataset variants.
    Vectors,
    /// The Silva dataset.
    Genomic,
    /// The RadioML dataset.
    RadioML,
}

impl Datasets {
    /// Create a new dataset variant from a string.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The name of the dataset.
    ///
    /// # Errors
    ///
    /// * If the dataset is not valid.
    fn from_str(dataset: &str) -> Result<Self, String> {
        let data_lower = dataset.to_lowercase();
        if data_lower.contains("radio") {
            Ok(Self::RadioML)
        } else if data_lower.contains("genomic") || data_lower.contains("silva") {
            Ok(Self::Genomic)
        } else {
            let data = ann_datasets::AnnDatasets::from_str(dataset);
            match data {
                Ok(_) => Ok(Self::Vectors),
                Err(e) => Err(e),
            }
        }
    }
}

/// Save the properties of all clusters in a single tree to a CSV file.
///
/// # Arguments
///
/// * `tree_dir` - The directory containing the tree.
/// * `metric` - The distance function to use.
/// * `is_expensive` - Whether the distance function is expensive.
/// * `output_dir` - The directory path to save the CSV file to.
///
/// # Errors
///
/// * If the tree directory does not exist or cannot be read.
/// * If the CSV file cannot be written to.
fn cakes_to_csv<I: Instance, U: Number, M: Instance>(
    tree_dir: &Path,
    metric: fn(&I, &I) -> U,
    is_expensive: bool,
    output_dir: &Path,
    dataset: &str,
) -> Result<(), String> {
    let cakes = Cakes::<I, U, VecDataset<I, U, M>>::load(tree_dir, metric, is_expensive)?;

    let cardinality = cakes.shard_cardinalities()[0];
    let output_path = output_dir.join(format!("{dataset}-{cardinality}-clusters.csv"));

    let mut csv =
        csv::Writer::from_path(output_path).map_err(|e| format!("Could not open CSV file: {e}"))?;
    let column_headers = [
        "id",
        "depth",
        "offset",
        "cardinality",
        "is_leaf",
        "radius",
        "lfd",
        "polar_distance",
        "ratio_cardinality",
        "ratio_radius",
        "ratio_lfd",
        "ratio_cardinality_ema",
        "ratio_radius_ema",
        "ratio_lfd_ema",
    ];
    csv.write_record(column_headers)
        .map_err(|e| format!("Could not write column headers to CSV file: {e}"))?;

    for (i, cluster) in cakes.trees()[0].root().subtree().into_iter().enumerate() {
        let mut row = vec![
            i.to_string(),
            cluster.depth().to_string(),
            cluster.offset().to_string(),
            cluster.cardinality().to_string(),
            cluster.is_leaf().to_string(),
            cluster.radius().to_string(),
            cluster.lfd().to_string(),
            cluster.polar_distance().unwrap_or_else(U::zero).to_string(),
        ];
        row.extend(
            cluster
                .ratios()
                .unwrap_or([0.0; 6])
                .iter()
                .map(ToString::to_string),
        );

        csv.write_record(row)
            .map_err(|e| format!("Could not write row {i} to CSV file: {e}"))?;
    }

    csv.flush()
        .map_err(|e| format!("Could not flush CSV file: {e}"))?;

    Ok(())
}
