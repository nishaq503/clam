//! Anomaly detection algorithms using CLAM.
use crate::{DistanceValue, Tree, chaoda::Node};

use super::{AnomalyFeatures, Graph};

mod accumulated_cardinality_ratios;
mod graph_neighborhood_size;
mod relative_cluster_cardinality;
mod relative_component_cardinality;
mod relative_vertex_degree;
mod stationary_probabilities;

pub use accumulated_cardinality_ratios::AccumulatedCardinalityRatios;
pub use graph_neighborhood_size::GraphNeighborhoodSize;
pub use relative_cluster_cardinality::RelativeClusterCardinality;
pub use relative_component_cardinality::RelativeComponentCardinality;
pub use relative_vertex_degree::RelativeVertexDegree;
pub use stationary_probabilities::StationaryProbabilities;

/// An anomaly detection algorithm that can be applied to a Chaoda graph.
///
/// Implementors of this trait should provide the [`Self::raw_anomaly_scores`] method and users should use the [`Self::anomaly_scores`] method to get normalized
/// anomaly scores in the range [0, 1] with higher scores indicating more anomalous items.
pub trait GraphAlgorithm<Id, I, T, A, M>: Send + Sync
where
    T: DistanceValue,
{
    /// Creates a new instance of the algorithm.
    fn new_boxed() -> Box<dyn GraphAlgorithm<Id, I, T, A, M>>
    where
        Self: Sized;

    /// Rank the nodes in the graph based on how anomalous they are, with `1` being the lowest rank and lower ranks indicating more anomalous nodes.
    fn rank_nodes<'a>(&self, graph: &'a Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<(&'a Node<T>, usize)>;

    /// Use the rankings of nodes to assign ranks to the items.
    fn rank_items(&self, graph: &Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<usize> {
        // TODO(Najib): Test this function to see if it correctly handles ties.

        let node_ranks = {
            let mut node_ranks = self.rank_nodes(graph, tree);
            node_ranks.sort_unstable_by_key(|(_, rank)| *rank);
            node_ranks
        };
        let mut ranks = vec![0; tree.cardinality()];
        let mut current_rank = 1;
        let mut previous_rank = 0;
        for (node, rank) in node_ranks {
            if rank != previous_rank {
                previous_rank = current_rank;
                current_rank = rank;
            }
            for item_id in node.iter_items() {
                ranks[item_id] = previous_rank;
            }
        }
        ranks
    }

    /// Parallel version of [`GraphAlgorithm::rank_nodes`], with the default implementation offering no parallelism.
    fn par_rank_nodes<'a>(&self, graph: &'a Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<(&'a Node<T>, usize)>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        self.rank_nodes(graph, tree)
    }

    /// Parallel version of [`GraphAlgorithm::rank_items`].
    fn par_rank_items(&self, graph: &Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<usize>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        let node_ranks = {
            let mut node_ranks = self.par_rank_nodes(graph, tree);
            node_ranks.sort_unstable_by_key(|(_, rank)| *rank);
            node_ranks
        };
        let mut ranks = vec![0; tree.cardinality()];
        let mut current_rank = 1;
        let mut previous_rank = 0;
        for (node, rank) in node_ranks {
            if rank != previous_rank {
                previous_rank = current_rank;
                current_rank = rank;
            }
            for item_id in node.iter_items() {
                ranks[item_id] = previous_rank;
            }
        }
        ranks
    }
}
