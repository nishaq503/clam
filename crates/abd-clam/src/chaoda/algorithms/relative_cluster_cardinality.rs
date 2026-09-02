//! A `Node` is more anomalous if it represents a number of items relative to other `Node`s in the `Graph`.

use crate::{DistanceValue, NamedAlgorithm, Tree, chaoda::Node};

use super::{AnomalyFeatures, Graph, GraphAlgorithm};

/// Assign anomaly scores to nodes based on the relative cardinality to other nodes in the graph.
#[derive(Debug, Clone)]
#[must_use]
pub struct RelativeClusterCardinality;

impl_named_algorithm_for_unit_struct!(RelativeClusterCardinality, "relative-cluster-cardinality", r"^relative-cluster-cardinality$");

impl<Id, I, T, A, M> GraphAlgorithm<Id, I, T, A, M> for RelativeClusterCardinality
where
    T: DistanceValue,
{
    fn new_boxed() -> Box<dyn GraphAlgorithm<Id, I, T, A, M>>
    where
        Self: Sized,
    {
        Box::new(Self)
    }

    fn rank_nodes<'a>(&self, graph: &'a Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<(&'a Node<T>, usize)> {
        // The more items in the node, the less anomalous it is, the higher the rank.
        graph.iter_nodes().map(|n| (n, n.num_items())).collect()
    }
}
