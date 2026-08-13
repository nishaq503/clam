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
    fn rank_nodes<'a>(&self, graph: &'a Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<(&'a Node<T>, usize)> {
        graph
            .iter_components()
            .flat_map(|c| {
                let score = c.iter_nodes().map(Node::num_items).sum::<usize>();
                c.iter_nodes().map(move |n| (n, score))
            })
            .collect::<Vec<_>>()
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
