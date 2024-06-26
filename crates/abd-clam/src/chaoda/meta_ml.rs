//! Meta Machine Learning models for CHAODA.

use std::path::Path;

use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use smartcore::{
    ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters},
    linear::{
        elastic_net::{ElasticNet, ElasticNetParameters},
        lasso::{Lasso, LassoParameters},
        linear_regression::{LinearRegression, LinearRegressionParameters},
        ridge_regression::{RidgeRegression, RidgeRegressionParameters},
    },
    tree::decision_tree_regressor::{DecisionTreeRegressor, DecisionTreeRegressorParameters},
};

/// A Meta Machine Learning model for CHAODA.
#[derive(Serialize, Deserialize)]
pub enum MlModel {
    /// A linear regression model.
    LinearRegression(LinearRegression<f32, f32, Array2<f32>, Vec<f32>>),
    /// An Elastic Net model.
    ElasticNet(ElasticNet<f32, f32, Array2<f32>, Vec<f32>>),
    /// A Lasso model.
    Lasso(Lasso<f32, f32, Array2<f32>, Vec<f32>>),
    /// A Ridge Regression model.
    RidgeRegression(RidgeRegression<f32, f32, Array2<f32>, Vec<f32>>),
    /// A Decision Tree Regressor model.
    DecisionTreeRegressor(DecisionTreeRegressor<f32, f32, Array2<f32>, Vec<f32>>),
    /// A Random Forest Regressor model.
    RandomForestRegressor(RandomForestRegressor<f32, f32, Array2<f32>, Vec<f32>>),
}

impl MlModel {
    /// Create a new `MetaMlModel`.
    ///
    /// # Arguments
    ///
    /// * `model`: The name of the model.
    ///
    /// # Errors
    ///
    /// * If the model name is unknown.
    pub fn new(model: &str) -> Result<Self, String> {
        let train_y = vec![0.0, 1.0, 0.5];
        let train_x = (0..10).flat_map(|_| train_y.clone()).collect::<Vec<_>>();
        let train_x = Array2::from_shape_vec((10, 3), train_x)
            .map_err(|e| e.to_string())?
            .t()
            .to_owned();
        // let train_x = DenseMatrix::new(3, 10, train_x, true).map_err(|e| e.to_string())?;
        println!("{model}: {train_x:?}");

        Ok(match model {
            "lr" | "LR" | "LinearRegression" => Self::LinearRegression(
                LinearRegression::fit(&train_x, &train_y, LinearRegressionParameters::default())
                    .map_err(|e| e.to_string())?,
            ),
            "en" | "EN" | "ElasticNet" => Self::ElasticNet(
                ElasticNet::fit(&train_x, &train_y, ElasticNetParameters::default()).map_err(|e| e.to_string())?,
            ),
            "la" | "LA" | "Lasso" => {
                Self::Lasso(Lasso::fit(&train_x, &train_y, LassoParameters::default()).map_err(|e| e.to_string())?)
            }
            "rr" | "RR" | "RidgeRegression" => Self::RidgeRegression(
                RidgeRegression::fit(&train_x, &train_y, RidgeRegressionParameters::default())
                    .map_err(|e| e.to_string())?,
            ),
            "dt" | "DT" | "DecisionTreeRegressor" => Self::DecisionTreeRegressor(
                DecisionTreeRegressor::fit(&train_x, &train_y, DecisionTreeRegressorParameters::default())
                    .map_err(|e| e.to_string())?,
            ),
            "rf" | "RF" | "RandomForestRegressor" => Self::RandomForestRegressor(
                RandomForestRegressor::fit(&train_x, &train_y, RandomForestRegressorParameters::default())
                    .map_err(|e| e.to_string())?,
            ),
            _ => return Err(format!("Unknown model: {model}")),
        })
    }

    /// Get the default models.
    #[must_use]
    pub fn defaults() -> Vec<Self> {
        let lr = Self::new("LR").unwrap_or_else(|e| unreachable!("{e}"));
        let dt = Self::new("DT").unwrap_or_else(|e| unreachable!("{e}"));
        vec![lr, dt]
    }

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
    pub fn train(&mut self, data: &Array2<f32>, roc_scores: &Vec<f32>) -> Result<(), String> {
        match self {
            Self::LinearRegression(model) => {
                *model = LinearRegression::fit(data, roc_scores, LinearRegressionParameters::default())
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
            Self::ElasticNet(model) => {
                *model =
                    ElasticNet::fit(data, roc_scores, ElasticNetParameters::default()).map_err(|e| e.to_string())?;
                Ok(())
            }
            Self::Lasso(model) => {
                *model = Lasso::fit(data, roc_scores, LassoParameters::default()).map_err(|e| e.to_string())?;
                Ok(())
            }
            Self::RidgeRegression(model) => {
                *model = RidgeRegression::fit(data, roc_scores, RidgeRegressionParameters::default())
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
            Self::DecisionTreeRegressor(model) => {
                *model = DecisionTreeRegressor::fit(data, roc_scores, DecisionTreeRegressorParameters::default())
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
            Self::RandomForestRegressor(model) => {
                *model = RandomForestRegressor::fit(data, roc_scores, RandomForestRegressorParameters::default())
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
        }
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
    /// * If the number of features in the data does not match the number of features in the model.
    /// * If the model cannot predict the data.
    pub fn predict(&self, data: &Array2<f32>) -> Result<Vec<f32>, String> {
        match self {
            Self::LinearRegression(model) => model.predict(data),
            Self::ElasticNet(model) => model.predict(data),
            Self::Lasso(model) => model.predict(data),
            Self::RidgeRegression(model) => model.predict(data),
            Self::DecisionTreeRegressor(model) => model.predict(data),
            Self::RandomForestRegressor(model) => model.predict(data),
        }
        .map_err(|e| e.to_string())
    }

    /// Save the model to a file.
    ///
    /// # Arguments
    ///
    /// * `path`: The path to the file.
    ///
    /// # Errors
    ///
    /// * If the file cannot be created.
    /// * If the model cannot be serialized.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        todo!("{path:?}")
        // let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        // bincode::serialize_into(file, self).map_err(|e| e.to_string())
    }

    /// Load the model from a file.
    ///
    /// # Arguments
    ///
    /// * `path`: The path to the file.
    ///
    /// # Errors
    ///
    /// * If the file cannot be opened.
    /// * If the model cannot be deserialized.
    pub fn load(path: &Path) -> Result<Self, String> {
        todo!("{path:?}")
        // let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        // bincode::deserialize_from(file).map_err(|e| e.to_string())
    }
}
