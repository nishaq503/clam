//! Meta Machine Learning models for CHAODA.

use std::path::Path;

use distances::Number;
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use smartcore::{
    api::{Predictor, SupervisedEstimator},
    ensemble::random_forest_regressor::RandomForestRegressor,
    linalg::basic::matrix::DenseMatrix,
    linear::linear_regression::LinearRegression,
    tree::decision_tree_regressor::DecisionTreeRegressor,
};

use super::OddBall;

// pub mod linear_regression;

/// A trait for a Meta Machine Learning model for CHAODA.
pub trait Model<U: Number, C: OddBall<U, 3>> {
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
    fn train<P: Clone + Default>(data: &DenseMatrix<f32>, roc_scores: &Array1<f32>) -> Result<Self, String>
    where
        Self: Sized + SupervisedEstimator<DenseMatrix<f32>, Array1<f32>, P>,
    {
        Self::fit(data, roc_scores, P::default()).map_err(|e| e.to_string())
    }

    /// Predict the suitability of a `Cluster` for selection in a `Graph`.
    ///
    /// # Arguments
    ///
    /// * `c`: The `Cluster` to predict the suitability of.
    ///
    /// # Returns
    ///
    /// The suitability of the `Cluster` for selection in a `Graph`.
    ///
    /// # Errors
    ///
    /// * If the prediction fails.
    fn predict(&self, c: &C) -> Result<f32, String>
    where
        Self: Predictor<DenseMatrix<f32>, Array1<f32>>,
    {
        let (p, p_) = c.properties();
        // join the properties
        let mut properties = p.to_vec();
        properties.extend_from_slice(&p_);

        // Create a DenseMatrix from the properties
        let properties = DenseMatrix::from_2d_array(&[properties.as_slice()]);

        // Predict the suitability of the Cluster
        let prediction = Predictor::predict(self, &properties).map_err(|e| e.to_string())?;
        Ok(prediction[0])
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
impl<U: Number, C: OddBall<U, 3>> Model<U, C> for LinearRegression<f32, f32, DenseMatrix<f32>, Array1<f32>> {}

/// A Meta Machine Learning model for CHAODA using an Elastic Net Regression model.
impl<U: Number, C: OddBall<U, 3>> Model<U, C>
    for smartcore::linear::elastic_net::ElasticNet<f32, f32, DenseMatrix<f32>, Array1<f32>>
{
}

/// A Meta Machine Learning model for CHAODA using a Lasso Regression model.
impl<U: Number, C: OddBall<U, 3>> Model<U, C>
    for smartcore::linear::lasso::Lasso<f32, f32, DenseMatrix<f32>, Array1<f32>>
{
}

/// A Meta Machine Learning model for CHAODA using a Ridge Regression model.
impl<U: Number, C: OddBall<U, 3>> Model<U, C>
    for smartcore::linear::ridge_regression::RidgeRegression<f32, f32, DenseMatrix<f32>, Array1<f32>>
{
}

/// A Meta Machine Learning model for CHAODA using a Decision Tree Regression model.
impl<U: Number, C: OddBall<U, 3>> Model<U, C> for DecisionTreeRegressor<f32, f32, DenseMatrix<f32>, Array1<f32>> {}

/// A Meta Machine Learning model for CHAODA using a Random Forest Regression model.
impl<U: Number, C: OddBall<U, 3>> Model<U, C> for RandomForestRegressor<f32, f32, DenseMatrix<f32>, Array1<f32>> {}
