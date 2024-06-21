//! Meta Machine Learning models for CHAODA.

use std::path::Path;

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::{Cluster, Dataset, Instance, Tree};

pub mod linear_regression;

/// A trait for a Meta Machine Learning model for CHAODA.
pub trait Model<'a, U: Number, C: Cluster<U>>: Sized + Serialize + for<'de> Deserialize<'de> {
    /// Train the model on a `Tree` and a set of labels.
    ///
    /// The number of `labels` must be equal to the cardinality of the data. The `labels`
    /// must be `true` for outliers and `false` for inliers.
    ///
    /// # Arguments
    ///
    /// * `tree`: The `Tree` to train the model on.
    /// * `labels`: The labels for the data.
    ///
    /// # Errors
    ///
    /// * If the number of `labels` is not equal to the cardinality of the data.
    fn train<I: Instance, D: Dataset<I, U>>(tree: &Tree<I, U, D, C>, labels: &[bool]) -> Result<Self, String>;

    /// Predict the suitability of a `Cluster` for selection in a `Graph`.
    fn predict(&self, c: &C) -> f64;

    /// Save the model to a file.
    ///
    /// # Errors
    ///
    /// * If the file cannot be created.
    /// * If the model cannot be serialized.
    fn save(&self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        bincode::serialize_into(file, self).map_err(|e| e.to_string())
    }

    /// Load the model from a file.
    ///
    /// # Errors
    ///
    /// * If the file cannot be opened.
    /// * If the model cannot be deserialized.
    fn load(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        bincode::deserialize_from(file).map_err(|e| e.to_string())
    }
}
