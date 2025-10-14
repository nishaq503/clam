//! Exporting the `Cluster` and its information to various formats.

use std::path::Path;

use crate::DistanceValue;

use super::Cluster;

/// The number of features to include in the CSV export.
const NUM_CLUSTER_FEATURES: usize = 7;

impl<T, A> Cluster<T, A>
where
    T: DistanceValue,
{
    /// Writes the `Cluster` tree to a CSV file at the specified path.
    ///
    /// # Errors
    ///
    /// - If the writer cannot be created.
    /// - If writing to the CSV file fails.
    /// - If flushing the writer fails.
    pub fn to_csv<P: AsRef<Path>>(&self, path: &P) -> std::io::Result<()> {
        let mut wtr = csv::Writer::from_path(path)?;
        wtr.write_record(Self::csv_header())?;

        for cluster in self.subtree_preorder() {
            wtr.write_record(cluster.csv_row())?;
        }

        wtr.flush()
    }

    /// Returns a CSV header string for the cluster information.
    const fn csv_header() -> [&'static str; NUM_CLUSTER_FEATURES] {
        [
            "center_index",
            "depth",
            "cardinality",
            "radius",
            "lfd",
            "span",
            "num_children",
        ]
    }

    /// Returns a row of CSV data representing the cluster's information.
    fn csv_row(&self) -> [String; NUM_CLUSTER_FEATURES] {
        [
            self.center_index.to_string(),
            self.depth.to_string(),
            self.cardinality().to_string(),
            self.radius().to_string(),
            self.lfd().to_string(),
            self.span().map_or_else(|| T::zero().to_string(), ToString::to_string),
            self.children().map_or(0, <[_]>::len).to_string(),
        ]
    }
}
