//! A trait to be implemented by machine learning regressors.

use std::path::Path;

use automl::IntoSupervisedData;
use ndarray::prelude::*;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;

/// Trait to represent types that can be used as a Meta-ML model
pub trait MetaMLModel {
    /// Train the model on the given features and targets.
    ///
    /// # Arguments
    ///
    /// * `dataset`: A `MetaMLDataset` for training.
    #[must_use]
    fn train(&mut self, dataset: MetaMLDataset) -> Self
    where
        Self: Sized;

    /// Makes a prediction given a trained model and 6 feature values.
    ///
    /// # Arguments
    ///
    /// * `features`: A reference to an array of 6 feature values.
    fn predict(&self, features: &[f32; 6]) -> f32;

    /// Loads a trained meta-ml model from disk.
    ///
    /// This function is used to load a previously trained meta-ml model from the specified file path.
    ///
    /// # Arguments
    ///
    /// * `path`: A reference to the file path where the model is stored.
    ///
    /// # Returns
    ///
    /// If successful, this function returns the loaded `MetaMLModel`.
    ///
    /// # Errors
    ///
    /// This function can return errors in the following cases:
    ///
    /// * If the serialized model cannot be read from the input file path.
    /// * If the trained model cannot be deserialized.
    fn load(path: &Path) -> Result<Self, String>
    where
        Self: Sized;

    /// Saves a trained meta-ml model to disk.
    ///
    /// # Arguments
    ///
    /// * `path`: A reference to the file path where the model will be saved.
    ///
    /// # Errors
    ///
    /// * If the model hasn't been trained.
    /// * If the trained model cannot be serialized.
    /// * If the serialized model cannot be written to the output file path.
    fn save(&self, path: &Path) -> Result<(), String>;
}

/// Represents the training data for a `MetaML` model
///
/// # Invariants:
///
/// * The number of rows in the features data is equal to the number of rows in the target data
/// * The data at row `i` of the features data corresponds to the data at row `i` of the targets data
pub struct MetaMLDataset {
    /// Features data for training the `MetaML` model.
    features: DenseMatrix<f32>,
    /// Target values for the corresponding features data.
    targets: Vec<f32>,
}

impl MetaMLDataset {
    /// Creates a dataset for training a meta-ml model from a set of feature
    /// values and their corresponding target values.
    ///
    /// # Arguments
    ///
    /// * `features`: A slice of arrays representing feature values, where each array has 6 elements.
    /// * `targets`: A slice of target values.
    ///
    /// # Returns
    ///
    /// A `MetaMLDataset` containing the provided feature and target data.
    ///
    /// # Errors
    ///
    /// * If the number of rows in the features data doesn't match the number of elements in the targets data.
    pub fn new(features: &Array2<f32>, targets: &Array1<f32>) -> Result<Self, String> {
        // TODO: better error checking once the rust branch is merged into master
        if features.nrows() == targets.len() {
            return Err("Different number of features and targets in input data".to_string());
        }

        let values = features.iter().copied().collect::<Vec<_>>();
        let features = DenseMatrix::from_array(features.nrows(), features.ncols(), &values);
        let targets = targets.to_vec();
        Ok(Self { features, targets })
    }

    /// Creates a dataset for training a meta-ml model from input data on disk.
    ///
    /// # Arguments
    ///
    /// * `features_file_path`: Path to the file containing feature data.
    /// * `targets_file_path`: Path to the file containing target data.
    ///
    /// # Returns
    ///
    /// Returns a `Result<Self, String>` where `Ok(Self)` indicates success, and `Err` contains an error message.
    ///
    /// # Errors
    ///
    /// * If either of the given files can't be found, opened, or parsed as `f32`s.
    /// * If the data contained within the features file isn't two-dimensional.
    /// * If the data contained within the targets file isn't one-dimensional.
    /// * If the number of rows in the features data doesn't match the number of elements in the targets data.
    pub fn from_npy(features_file_path: &Path, targets_file_path: &Path) -> Result<Self, String> {
        let features = ndarray_npy::read_npy(features_file_path).map_err(|e| e.to_string())?;
        let targets = ndarray_npy::read_npy(targets_file_path).map_err(|e| e.to_string())?;
        Self::new(&features, &targets)
    }
}

impl IntoSupervisedData for MetaMLDataset {
    /// Converts the current dataset into a tuple containing feature data and target data.
    ///
    /// This function transforms the dataset into a format suitable for supervised learning, returning
    /// a tuple where the first element is a two-dimensional feature matrix, and the second element is
    /// a one-dimensional target vector.
    ///
    /// # Returns
    ///
    /// A tuple containing feature data represented as a `DenseMatrix<f32>` and target data represented
    /// as a `Vec<f32>`.
    fn to_supervised_data(self) -> (DenseMatrix<f32>, Vec<f32>) {
        (self.features, self.targets)
    }
}
