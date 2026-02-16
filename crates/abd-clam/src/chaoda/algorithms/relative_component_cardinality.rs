//! A `Node` is more anomalous if it is in a `Component` whose nodes collectively have fewer items than the other `Components` in the graph.

use crate::{DistanceValue, NamedAlgorithm, Tree};

use super::{super::Node, AnomalyFeatures, Graph, GraphAlgorithm, ParGraphAlgorithm};

/// A `Node` is more anomalous if it is in a `Component` whose nodes collectively have fewer items than the other `Components` in the graph.
#[derive(Debug, Clone)]
#[must_use]
pub struct RelativeComponentCardinality;

impl_named_algorithm_for_unit_struct!(
    RelativeComponentCardinality,
    "relative-component-cardinality",
    r"^relative-component-cardinality$"
);

impl<Id, I, T, A, M> GraphAlgorithm<Id, I, T, A, M> for RelativeComponentCardinality
where
    T: DistanceValue,
{
    #[expect(clippy::cast_precision_loss)]
    fn raw_anomaly_scores(&self, graph: &Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<f64>, String> {
        let mut scores = graph
            .iter_components()
            .flat_map(|c| {
                // The more items in the component, the less anomalous its nodes are, thus the negative sign.
                let score = -(c.iter_nodes().map(Node::num_items).sum::<usize>() as f64);
                c.iter_nodes().flat_map(move |n| n.iter_items().map(move |i| (i, score)))
            })
            .collect::<Vec<_>>();
        scores.sort_by_key(|(i, _)| *i);
        Ok(scores.into_iter().map(|(_, score)| score).collect())
    }
}

impl<Id, I, T, A, M> ParGraphAlgorithm<Id, I, T, A, M> for RelativeComponentCardinality
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
}
