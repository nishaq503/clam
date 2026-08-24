//! Meta-Machine Learning (Meta-ML) models for use in `CHAODA`.

use crate::Cluster;

use super::AnomalyFeatures;

mod decision_tree;
mod linear_regression;

pub use decision_tree::DecisionTree;
pub use linear_regression::LinearRegression;

/// A trait for models that can predict the quality of a `Cluster` for use in a `Graph` based on its anomaly features.
///
/// Higher scores indicate that the `Cluster` should be used in the `Graph`, while lower scores indicate that it should not be used.
pub trait MetaMlPredictor<T, A>: Send + Sync {
    /// Predict the quality of a `Cluster` for use in a `Graph` based on its anomaly features.
    ///
    /// # Errors
    ///
    /// - If the model fails to make a prediction.
    fn predict(&self, cluster: &Cluster<T, (A, AnomalyFeatures)>) -> Result<f64, String>;
}

/// A trait for models that can be trained to predict the quality of a `Cluster` for use in a `Graph` based on its anomaly features.
pub trait MetaMlTrainer<T, A>: MetaMlPredictor<T, A> + Sized {
    /// Train the model on the given training data.
    ///
    /// # Errors
    ///
    /// - If the training data is invalid or the model cannot be trained.
    fn fit(data: &[(AnomalyFeatures, f64)]) -> Result<Self, String>;
}

/// A simple meta-ml model that selects clusters of a given depth in the tree.
#[must_use]
#[derive(Debug, Clone)]
pub struct Layer {
    /// The depth of the clusters to select.
    depth: usize,
}

impl Layer {
    /// Create a new `Layer` model that selects clusters of the given depth.
    pub const fn new(depth: usize) -> Self {
        Self { depth }
    }
}

impl<T, A> MetaMlPredictor<T, A> for Layer {
    fn predict(&self, cluster: &Cluster<T, (A, AnomalyFeatures)>) -> Result<f64, String> {
        let y = if cluster.depth == self.depth { 1.0 } else { 0.0 };
        Ok(y)
    }
}
