//! Meta-Machine Learning (Meta-ML) models for use in `CHAODA`.

use crate::Cluster;

use super::AnomalyFeatures;

/// A trait for models that can predict the quality of a `Cluster` for use in a `Graph` based on its anomaly features.
///
/// Higher scores indicate that the `Cluster` should be used in the `Graph`, while lower scores indicate that it should not be used.
pub trait MetaMlPredictor<T, A> {
    /// Predict the quality of a `Cluster` for use in a `Graph` based on its anomaly features.
    fn predict(&self, cluster: &Cluster<T, (A, AnomalyFeatures)>) -> f64;
}

/// A trait for models that can be trained to predict the quality of a `Cluster` for use in a `Graph` based on its anomaly features.
pub trait MetaMlTrainer<T, A>: MetaMlPredictor<T, A> {
    /// Train the model on the given training data.
    ///
    /// # Errors
    ///
    /// - If the training data is invalid or the model cannot be trained.
    fn fit(&mut self, clusters: &[Cluster<T, (A, AnomalyFeatures)>], scores: &[f64]) -> Result<(), String>;
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
    fn predict(&self, cluster: &Cluster<T, (A, AnomalyFeatures)>) -> f64 {
        if cluster.depth == self.depth { 1.0 } else { 0.0 }
    }
}

/// A linear regression model.
#[must_use]
#[derive(Debug, Clone)]
pub struct LinearRegression {
    /// The weights of the model.
    ///
    /// The length of the weights vector should be equal to the number of features in the anomaly features.
    weights: Vec<f64>,
    /// The bias of the model.
    bias: f64,
}

impl LinearRegression {
    /// Create a new `LinearRegression` model with the given number of features.
    pub fn new() -> Self {
        let n_features = 6; // The number of features in the AnomalyFeatures struct.
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
        }
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, A> MetaMlPredictor<T, A> for LinearRegression {
    fn predict(&self, cluster: &Cluster<T, (A, AnomalyFeatures)>) -> f64 {
        let features = cluster.annotation().1.as_vec();
        let mut score = self.bias;
        for (i, weight) in self.weights.iter().enumerate() {
            score += weight * features[i];
        }
        score
    }
}

impl<T, A> MetaMlTrainer<T, A> for LinearRegression {
    #[expect(unused_variables)]
    fn fit(&mut self, clusters: &[Cluster<T, (A, AnomalyFeatures)>], scores: &[f64]) -> Result<(), String> {
        todo!("Use a linear regression library to fit the model to the training data.");
    }
}

/// A decision tree model.
#[must_use]
#[derive(Debug, Clone)]
pub struct DecisionTree;

impl<T, A> MetaMlPredictor<T, A> for DecisionTree {
    #[expect(unused_variables)]
    fn predict(&self, cluster: &Cluster<T, (A, AnomalyFeatures)>) -> f64 {
        todo!("Use a decision tree library to predict the score of the cluster.");
    }
}

impl<T, A> MetaMlTrainer<T, A> for DecisionTree {
    #[expect(unused_variables)]
    fn fit(&mut self, clusters: &[Cluster<T, (A, AnomalyFeatures)>], scores: &[f64]) -> Result<(), String> {
        todo!("Use a decision tree library to fit the model to the training data.");
    }
}
