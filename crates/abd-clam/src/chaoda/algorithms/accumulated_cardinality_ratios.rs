//! A `Node` is more anomalous if it comes from a cluster whose accumulated cardinality ratio is low.

use crate::{DistanceValue, NamedAlgorithm, Tree, chaoda::Node};

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
    fn rank_nodes<'a>(&self, graph: &'a Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<(&'a Node<T>, usize)> {
        let mut scores = graph
            .iter_nodes()
            .map(|n| (n, tree.get_cluster_unchecked(n.direct_center_index()).annotation.1.cardinality_ratio))
            .collect::<Vec<_>>();
        scores.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        scores.into_iter().enumerate().map(|(rank, (node, _))| (node, rank)).collect()
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
