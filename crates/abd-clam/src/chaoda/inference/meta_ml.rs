//! Inferring with meta-ml models.

use serde::{Deserialize, Serialize};
use smartcore::{
    linalg::basic::matrix::DenseMatrix, linear::linear_regression::LinearRegression,
    tree::decision_tree_regressor::DecisionTreeRegressor,
};

use crate::chaoda::NUM_RATIOS;

use super::py_models;

/// A trained meta-ml model.
#[derive(Serialize, Deserialize)]
pub enum TrainedMetaMlModel {
    /// A linear regression model.
    LinearRegression(LinearRegression<f32, f32, DenseMatrix<f32>, Vec<f32>>),
    /// A Decision Tree model.
    DecisionTree(DecisionTreeRegressor<f32, f32, DenseMatrix<f32>, Vec<f32>>),
    /// Pre-trained liner regression model from python we used for the CHAODA paper.
    #[serde(skip)]
    PyLr(fn([f32; NUM_RATIOS]) -> f32),
    /// Pre-trained Decision Tree model from python we used for the CHAODA paper.
    #[serde(skip)]
    PyDt(fn([f32; NUM_RATIOS]) -> f32),
}

impl TrainedMetaMlModel {
    /// Get the name of the model.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::LinearRegression(_) => "LinearRegression",
            Self::DecisionTree(_) => "DecisionTree",
            Self::PyLr(_) => "PyLr",
            Self::PyDt(_) => "PyDt",
        }
    }

    /// Get a short name for the model.
    #[must_use]
    pub const fn short_name(&self) -> &str {
        match self {
            Self::LinearRegression(_) => "LR",
            Self::DecisionTree(_) => "DT",
            Self::PyLr(_) => "PyLr",
            Self::PyDt(_) => "PyDt",
        }
    }

    /// Predict the suitability of several `Cluster`s for selection in a `Graph`.
    ///
    /// This method is convenient when we want to predict the suitability of several `Cluster`s at once,
    /// and using several `MetaML` models.
    ///
    /// # Arguments
    ///
    /// * `props`: A matrix where each row contains the aggregated anomaly properties of `Cluster`s in a `Graph`.
    ///
    /// # Returns
    ///
    /// The suitability of the `Cluster`s for selection in a `Graph`.
    ///
    /// # Errors
    ///
    /// * If the number of features in the data does not match the number of features in the model.
    /// * If the model cannot predict the data.
    pub fn predict(&self, props: &[f32]) -> Result<Vec<f32>, String> {
        if props.is_empty() || (props.len() % NUM_RATIOS != 0) {
            return Err(format!(
                "Number of features in data ({}) does not match number of features in model ({})",
                props.len(),
                NUM_RATIOS
            ));
        }
        let props_vec = props.chunks_exact(NUM_RATIOS).map(<[f32]>::to_vec).collect::<Vec<_>>();
        let props_matrix =
            DenseMatrix::from_2d_vec(&props_vec).map_err(|e| format!("Failed to create matrix of samples: {e}"))?;

        match self {
            Self::LinearRegression(model) => model
                .predict(&props_matrix)
                .map_err(|e| format!("Failed to predict with LinearRegression model: {e}")),
            Self::DecisionTree(model) => model
                .predict(&props_matrix)
                .map_err(|e| format!("Failed to predict with DecisionTree model: {e}")),
            Self::PyLr(model) => Ok(props
                .chunks_exact(NUM_RATIOS)
                .map(|p| model([p[0], p[1], p[2], p[3], p[4], p[5]]))
                .collect()),
            Self::PyDt(model) => Ok(props
                .chunks_exact(NUM_RATIOS)
                .map(|p| model([p[0], p[1], p[2], p[3], p[4], p[5]]))
                .collect()),
        }
    }

    /// Load a pre-trained model that was trained in python.
    ///
    /// # Arguments
    ///
    /// * `meta_name`: The name of the meta-ml model. Must be one of `["lr", "dt"]`.
    /// * `metric_name`: The name of the distance metric used to train the model. Must be one of `["euclidean", "manhattan"]`.
    /// * `member_name`: The name of the member of the ensemble. Must be one of `["sc", "cc", "gn", "cr", "sp", "vd"]`.
    ///
    /// # Errors
    ///
    /// * If the model cannot be loaded. This can only happen if any of the names are incorrect.
    pub fn load_py(meta_name: &str, metric_name: &str, member_name: &str) -> Result<Self, String> {
        let name = format!("{meta_name}_{metric_name}_{member_name}");
        let model = py_models::get_py_model(&name)?;

        match meta_name {
            "lr" => Ok(Self::PyLr(model)),
            "dt" => Ok(Self::PyDt(model)),
            _ => Err(format!("Unknown meta-ml model: {meta_name}")),
        }
    }
}
