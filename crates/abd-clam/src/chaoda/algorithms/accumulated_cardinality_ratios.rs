//! A `Node` is more anomalous if it comes from a cluster whose accumulated cardinality ratio is low.

use crate::{DistanceValue, NamedAlgorithm, Tree};

use super::{AnomalyFeatures, Graph, GraphAlgorithm, ParGraphAlgorithm};

/// Assign anomaly scores to nodes based on the accumulated cardinality ratios of their clusters.
#[derive(Debug, Clone)]
#[must_use]
pub struct AccumulatedCardinalityRatios;

impl_named_algorithm_for_unit_struct!(
    AccumulatedCardinalityRatios,
    "accumulated-cardinality-ratios",
    r"^accumulated-cardinality-ratios$"
);

impl<Id, I, T, A, M> GraphAlgorithm<Id, I, T, A, M> for AccumulatedCardinalityRatios
where
    T: DistanceValue,
{
    fn raw_anomaly_scores(&self, graph: &Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<f64>, String> {
        let clusters = graph
            .iter_nodes()
            .map(|n| tree.get_cluster_unchecked(n.direct_center_index()))
            .map(|c| (c, c.annotation.1.cardinality_ratio))
            .collect::<Vec<_>>();

        let mut scores = clusters
            .into_iter()
            .flat_map(|(c, score)| c.items_range().map(move |i| (i, score)))
            .collect::<Vec<_>>();
        scores.sort_by_key(|(i, _)| *i);
        Ok(scores.into_iter().map(|(_, score)| score).collect())
    }
}

impl<Id, I, T, A, M> ParGraphAlgorithm<Id, I, T, A, M> for AccumulatedCardinalityRatios
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
}
