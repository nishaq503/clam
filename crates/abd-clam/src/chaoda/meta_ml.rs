//! Meta Machine Learning models for CHAODA.

use std::path::Path;

use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use smartcore::{
    api::{Predictor, SupervisedEstimator},
    ensemble::random_forest_regressor::RandomForestRegressor,
    linalg::basic::matrix::DenseMatrix,
    linear::{
        elastic_net::ElasticNet, lasso::Lasso, linear_regression::LinearRegression, ridge_regression::RidgeRegression,
    },
    tree::decision_tree_regressor::DecisionTreeRegressor,
};

/// A trait for a Meta Machine Learning model for CHAODA.
pub trait Model {
    /// Train the model on data from a `Graph`.
    ///
    /// # Arguments
    ///
    /// * `data`: A matrix where each row contains the aggregated anomaly properties of `Cluster`s in a `Graph`.
    /// * `roc_scores`: The ROC score for each `Cluster`.
    ///
    /// # Errors
    ///
    /// * If the number of `labels` is not equal to the cardinality of the data.
    ///
    /// # Generics
    ///
    /// * `P`: The type of the parameters of the model.
    fn train<P: Clone + Default>(data: &DenseMatrix<f32>, roc_scores: &Array1<f32>) -> Result<Self, String>
    where
        Self: Sized + SupervisedEstimator<DenseMatrix<f32>, Array1<f32>, P>,
    {
        Self::fit(data, roc_scores, P::default()).map_err(|e| e.to_string())
    }

    /// Predict the suitability of several `Cluster`s for selection in a `Graph`.
    ///
    /// This method is convenient when we want to predict the suitability of several `Cluster`s at once,
    /// and using several `MetaML` models.
    ///
    /// # Arguments
    ///
    /// * `data`: A matrix where each row contains the aggregated anomaly properties of `Cluster`s in a `Graph`.
    ///
    /// # Returns
    ///
    /// The suitability of the `Cluster`s for selection in a `Graph`.
    ///
    /// # Errors
    ///
    /// * If the prediction fails.
    fn predict(&self, data: &DenseMatrix<f32>) -> Result<Array1<f32>, String>
    where
        Self: Predictor<DenseMatrix<f32>, Array1<f32>>,
    {
        Predictor::predict(self, data).map_err(|e| e.to_string())
    }

    /// Save the model to a file.
    ///
    /// # Errors
    ///
    /// * If the file cannot be created.
    /// * If the model cannot be serialized.
    fn save(&self, path: &Path) -> Result<(), String>
    where
        Self: Serialize,
    {
        let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        bincode::serialize_into(file, self).map_err(|e| e.to_string())
    }

    /// Load the model from a file.
    ///
    /// # Errors
    ///
    /// * If the file cannot be opened.
    /// * If the model cannot be deserialized.
    fn load(path: &Path) -> Result<Self, String>
    where
        Self: for<'de> Deserialize<'de>,
    {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        bincode::deserialize_from(file).map_err(|e| e.to_string())
    }
}

/// A Meta Machine Learning model for CHAODA using a Linear Regression model.
impl Model for LinearRegression<f32, f32, DenseMatrix<f32>, Array1<f32>> {}

/// A Meta Machine Learning model for CHAODA using an Elastic Net Regression model.
impl Model for ElasticNet<f32, f32, DenseMatrix<f32>, Array1<f32>> {}

/// A Meta Machine Learning model for CHAODA using a Lasso Regression model.
impl Model for Lasso<f32, f32, DenseMatrix<f32>, Array1<f32>> {}

/// A Meta Machine Learning model for CHAODA using a Ridge Regression model.
impl Model for RidgeRegression<f32, f32, DenseMatrix<f32>, Array1<f32>> {}

/// A Meta Machine Learning model for CHAODA using a Decision Tree Regression model.
impl Model for DecisionTreeRegressor<f32, f32, DenseMatrix<f32>, Array1<f32>> {}

/// A Meta Machine Learning model for CHAODA using a Random Forest Regression model.
impl Model for RandomForestRegressor<f32, f32, DenseMatrix<f32>, Array1<f32>> {}
