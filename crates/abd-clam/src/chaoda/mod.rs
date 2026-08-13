//! Anomaly detection algorithms using CLAM.
//!
//! This module contains CLAM-CHAODA (Clustered Hierarchical Anomaly and Outlier Detection Algorithms). This is a family of algorithms that use CLAM trees to
//! impute graphs that enable unsupervised anomaly detection algorithms.
//!
//! For trees, this enables the [`Tree::annotate_anomaly_features`](crate::Tree::annotate_anomaly_features) method (along with its parallel version). These
//! features can then be used for creating CHAODA graphs. The graphs can, in turn, be used for anomaly detection using the algorithms we provide...

mod features;
mod graph;
pub mod meta_ml;
mod training;
mod tree;

pub mod algorithms;

pub use features::{AnomalyFeatures, normalize_features};
pub use graph::{Component, Graph, Node};

/// Compute the ROC AUC score for binary classification.
///
/// # Arguments
///
/// * `y_true` - An iterator of boolean values representing the true binary labels.
/// * `y_pred` - A vector of floating-point values representing the predicted probabilities for the positive class.
///
/// # Returns
///
/// The ROC AUC score as a floating-point value between 0.0 and 1.0.
///
/// # Errors
///
/// - If the lengths of `y_true` and `y_pred` do not match.
pub fn roc_auc_score<Ids>(y_true: Ids, y_pred: &Vec<f64>) -> Result<f64, String>
where
    Ids: Iterator<Item = bool>,
{
    let y_true = y_true.map(|b| if b { 1.0 } else { 0.0 }).collect::<Vec<_>>();
    if y_true.len() == y_pred.len() {
        Ok(smartcore::metrics::roc_auc_score(&y_true, y_pred))
    } else {
        Err(format!(
            "Length mismatch: y_true has length {}, but y_pred has length {}",
            y_true.len(),
            y_pred.len()
        ))
    }
}
