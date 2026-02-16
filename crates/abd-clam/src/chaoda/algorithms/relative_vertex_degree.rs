//! A `Node` is more anomalous if it has fewer neighbors in the graph relative to other nodes in the graph.

use crate::{DistanceValue, NamedAlgorithm, Tree};

use super::{AnomalyFeatures, Graph, GraphAlgorithm, ParGraphAlgorithm};

/// A `Node` is more anomalous if it has fewer neighbors in the graph relative to other nodes in the graph.
#[derive(Debug, Clone)]
#[must_use]
pub struct RelativeVertexDegree;

impl_named_algorithm_for_unit_struct!(RelativeVertexDegree, "relative-vertex-degree", r"^relative-vertex-degree$");

impl<Id, I, T, A, M> GraphAlgorithm<Id, I, T, A, M> for RelativeVertexDegree
where
    T: DistanceValue,
{
    #[expect(clippy::cast_precision_loss)]
    fn raw_anomaly_scores(&self, graph: &Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<f64>, String> {
        let mut scores = graph
            .iter_nodes()
            .flat_map(|n| {
                // The more neighbors a node has, the less anomalous it is, thus the negative sign.
                let score = -(n.num_edges() as f64);
                n.iter_items().map(move |i| (i, score))
            })
            .collect::<Vec<_>>();
        scores.sort_by_key(|(i, _)| *i);
        Ok(scores.into_iter().map(|(_, score)| score).collect())
    }
}

impl<Id, I, T, A, M> ParGraphAlgorithm<Id, I, T, A, M> for RelativeVertexDegree
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
}
