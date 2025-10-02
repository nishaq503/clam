//! Exporting the `Cluster` and its information to various formats.

use std::{fmt::Display, path::Path};

use crate::{Cluster, DistanceValue};

/// The number of features to include in the CSV export.
const NUM_CLUSTER_FEATURES: usize = 8;

impl<Id: Display, I, T: DistanceValue, A> Cluster<Id, I, T, A> {
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

        for cluster in self.subtree() {
            wtr.write_record(cluster.csv_row())?;
        }

        wtr.flush()
    }

    /// Returns a CSV header string for the cluster information.
    const fn csv_header() -> [&'static str; NUM_CLUSTER_FEATURES] {
        [
            "center_id",
            "depth",
            "cardinality",
            "radius",
            "lfd",
            "radial_sum",
            "span",
            "num_children",
        ]
    }

    /// Returns a row of CSV data representing the cluster's information.
    fn csv_row(&self) -> [String; NUM_CLUSTER_FEATURES] {
        [
            self.center.0.to_string(),
            self.depth.to_string(),
            self.cardinality().to_string(),
            self.radius().to_string(),
            self.lfd().to_string(),
            self.radial_sum().to_string(),
            self.span.to_string(),
            self.children().map_or(0, <[_]>::len).to_string(),
        ]
    }
}
