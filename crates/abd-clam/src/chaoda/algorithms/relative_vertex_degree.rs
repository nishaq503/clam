//! A `Node` is more anomalous if it has fewer neighbors in the graph relative to other nodes in the graph.

use crate::{DistanceValue, NamedAlgorithm, Tree, chaoda::Node};

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
    fn rank_nodes<'a>(&self, graph: &'a Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<(&'a Node<T>, usize)> {
        // The more edges a node has, the less anomalous it is, the higher the rank.
        graph.iter_nodes().map(|n| (n, n.num_edges())).collect()
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
