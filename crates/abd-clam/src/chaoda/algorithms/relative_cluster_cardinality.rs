//! A `Node` is more anomalous if it represents a number of items relative to other `Node`s in the `Graph`.

use crate::{DistanceValue, NamedAlgorithm, Tree};

use super::{AnomalyFeatures, Graph, GraphAlgorithm, ParGraphAlgorithm};

/// Assign anomaly scores to nodes based on the relative cardinality to other nodes in the graph.
#[derive(Debug, Clone)]
#[must_use]
pub struct RelativeClusterCardinality;

impl_named_algorithm_for_unit_struct!(RelativeClusterCardinality, "relative-cluster-cardinality", r"^relative-cluster-cardinality$");

impl<Id, I, T, A, M> GraphAlgorithm<Id, I, T, A, M> for RelativeClusterCardinality
where
    T: DistanceValue,
{
    #[expect(clippy::cast_precision_loss)]
    fn raw_anomaly_scores(&self, graph: &Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<f64>, String> {
        let mut scores = graph
            .iter_nodes()
            .flat_map(|n| {
                // The more items in the node, the less anomalous it is, thus the negative sign.
                let score = -(n.num_items() as f64);
                n.iter_items().map(move |i| (i, score))
            })
            .collect::<Vec<_>>();
        scores.sort_by_key(|(i, _)| *i);
        Ok(scores.into_iter().map(|(_, score)| score).collect())
    }
}

impl<Id, I, T, A, M> ParGraphAlgorithm<Id, I, T, A, M> for RelativeClusterCardinality
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
}
