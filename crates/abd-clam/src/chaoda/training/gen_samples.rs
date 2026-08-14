//! Functions for generating training samples to train meta-ML algorithms.

use crate::{DistanceValue, Tree};

use super::super::{AnomalyFeatures, Graph, algorithms, roc_auc_score};

/// Generates a single training sample from a given `Tree`, `Graph`, and `Chaoda` algorithm.
///
/// The features for the training sample are the mean anomaly features of the `Cluster`s in the `tree` that were selected for creating the `Graph`. The target
/// variable for the training sample is the ROC AUC score of the anomaly scores computed by the `algorithm` for the `tree` and `graph`.
///
/// # Arguments
///
/// - `tree`: The `Tree` used for creating the `Graph`.
/// - `graph`: The `Graph` for which the training sample is being generated.
/// - `algorithm`: The `Chaoda` algorithm used to compute anomaly scores for the `tree` and `graph`.
///
/// # Returns
///
/// A tuple containing:
///
/// - The mean of the anomaly features of the `Cluster`s in the `tree` that were selected for creating the `Graph`.
/// - The ROC AUC score of the anomaly scores computed by the `algorithm` for the `tree` and `graph`.
///
/// # Errors
///
/// - If the `algorithm` fails to compute anomaly scores for the `tree` and `graph`. See [`Chaoda::anomaly_scores`] for more details on possible errors.
/// - If the ROC AUC score cannot be computed from the true labels and predicted scores. See [`roc_auc_score`] for more details on possible errors.
pub fn gen_training_sample<Id, I, T, A, M, Alg, Oracle>(
    tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>,
    graph: &Graph<T>,
    algorithm: &Alg,
    oracle: &Oracle,
) -> Result<(AnomalyFeatures, f64), String>
where
    T: DistanceValue,
    Alg: AsRef<dyn algorithms::GraphAlgorithm<Id, I, T, A, M>>,
    Oracle: Fn(&Id) -> bool,
{
    let y_pred = algorithm.as_ref().anomaly_scores(graph, tree);

    let y_true = tree.items.iter().map(|(a, _, _)| oracle(a)).collect::<Vec<_>>();
    let auc = roc_auc_score(&y_true, &y_pred)?;

    Ok((graph.mean_anomaly_features(tree), auc))
}
