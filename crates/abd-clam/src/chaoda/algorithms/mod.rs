//! Anomaly detection algorithms using CLAM.

#![expect(dead_code)]

use crate::{DistanceValue, NamedAlgorithm, Tree, chaoda::Node};

use super::{AnomalyFeatures, Graph};

mod accumulated_cardinality_ratios;
mod graph_neighborhood_size;
mod relative_cluster_cardinality;
mod relative_component_cardinality;
mod relative_vertex_degree;
mod stationary_probabilities;

use accumulated_cardinality_ratios::AccumulatedCardinalityRatios;
use graph_neighborhood_size::GraphNeighborhoodSize;
use relative_cluster_cardinality::RelativeClusterCardinality;
use relative_component_cardinality::RelativeComponentCardinality;
use relative_vertex_degree::RelativeVertexDegree;
use stationary_probabilities::StationaryProbabilities;

/// All anomaly detection algorithms provided with CHAODA.
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub enum ChaodaAlgorithm {
    /// A `Node` is more anomalous if it comes from a cluster whose accumulated cardinality ratio is low.
    AccumulatedCardinalityRatios(AccumulatedCardinalityRatios),
    /// A `Node` is more anomalous if it can reach fewer other nodes in the graph within the same number of steps as compared to other nodes in the graph.
    GraphNeighborhoodSize(GraphNeighborhoodSize),
    /// A `Node` is more anomalous if it represents a smaller number of items relative to other `Node`s in the `Graph`.
    RelativeClusterCardinality(RelativeClusterCardinality),
    /// A `Node` is more anomalous if it is in a `Component` whose nodes collectively have fewer items than the other `Components` in the graph.
    RelativeComponentCardinality(RelativeComponentCardinality),
    /// A `Node` is more anomalous if it has fewer neighbors in the graph relative to other nodes in the graph.
    RelativeVertexDegree(RelativeVertexDegree),
    /// A `Node` is more anomalous if it is visited less frequently during an infinite random walk on the graph.
    StationaryProbabilities(StationaryProbabilities),
}

impl core::fmt::Display for ChaodaAlgorithm {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AccumulatedCardinalityRatios(alg) => alg.fmt(f),
            Self::GraphNeighborhoodSize(alg) => alg.fmt(f),
            Self::RelativeClusterCardinality(alg) => alg.fmt(f),
            Self::RelativeComponentCardinality(alg) => alg.fmt(f),
            Self::RelativeVertexDegree(alg) => alg.fmt(f),
            Self::StationaryProbabilities(alg) => alg.fmt(f),
        }
    }
}

impl core::str::FromStr for ChaodaAlgorithm {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::regex_pattern().captures(s).map_or_else(
            || Err(format!("Invalid format for ChaodaAlgorithm: {s}")),
            |caps| {
                let algorithm = caps.get(1).map(|m| m.as_str());
                match algorithm {
                    Some("accumulated-cardinality-ratios") => AccumulatedCardinalityRatios::from_str(s).map(Self::AccumulatedCardinalityRatios),
                    Some("graph-neighborhood-size") => GraphNeighborhoodSize::from_str(s).map(Self::GraphNeighborhoodSize),
                    Some("relative-cluster-cardinality") => RelativeClusterCardinality::from_str(s).map(Self::RelativeClusterCardinality),
                    Some("relative-component-cardinality") => RelativeComponentCardinality::from_str(s).map(Self::RelativeComponentCardinality),
                    Some("relative-vertex-degree") => RelativeVertexDegree::from_str(s).map(Self::RelativeVertexDegree),
                    Some("stationary-probabilities") => StationaryProbabilities::from_str(s).map(Self::StationaryProbabilities),
                    Some(algorithm) => Err(format!("Unknown ChaodaAlgorithm algorithm: {algorithm}. Must be one of accumulated-cardinality-ratios, graph-neighborhood-size, relative-cluster-cardinality, relative-component-cardinality, relative-vertex-degree, or stationary-probabilities.")),
                    None => Err(format!("Invalid format for ChaodaAlgorithm: {s}")),
                }
            },
        )
    }
}

impl NamedAlgorithm for ChaodaAlgorithm {
    fn name(&self) -> &'static str {
        match self {
            Self::AccumulatedCardinalityRatios(alg) => alg.name(),
            Self::GraphNeighborhoodSize(alg) => alg.name(),
            Self::RelativeClusterCardinality(alg) => alg.name(),
            Self::RelativeComponentCardinality(alg) => alg.name(),
            Self::RelativeVertexDegree(alg) => alg.name(),
            Self::StationaryProbabilities(alg) => alg.name(),
        }
    }

    fn regex_pattern<'a>() -> &'a lazy_regex::Regex {
        lazy_regex::regex!(
            r"^(accumulated-cardinality-ratios|graph-neighborhood-size|relative-cluster-cardinality|relative-component-cardinality|relative-vertex-degree|stationary-probabilities)$"
        )
    }
}

impl ChaodaAlgorithm {
    /// Rank the items in the graph based on how anomalous they are, with `1` being the lowest rank and lower ranks indicating more anomalous items.
    ///
    /// # Arguments
    ///
    /// - `graph`: The `Graph` for which to compute anomaly scores.
    /// - `tree`: The `Tree` that was used for creating the `Graph`.
    ///
    /// # Errors
    ///
    /// - If any of the `Cluster`s selected for creating the `Graph` was not found in the `Tree`.
    /// - If the underlying algorithm fails to compute a score for each item in the tree.
    pub fn rank_items<Id, I, T, A, M>(&self, graph: &Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<usize>, String>
    where
        T: DistanceValue,
    {
        match self {
            Self::AccumulatedCardinalityRatios(alg) => alg.rank_items(graph, tree),
            Self::GraphNeighborhoodSize(alg) => alg.rank_items(graph, tree),
            Self::RelativeClusterCardinality(alg) => alg.rank_items(graph, tree),
            Self::RelativeComponentCardinality(alg) => alg.rank_items(graph, tree),
            Self::RelativeVertexDegree(alg) => alg.rank_items(graph, tree),
            Self::StationaryProbabilities(alg) => alg.rank_items(graph, tree),
        }
    }

    /// Parallel version of [`Self::rank_items`].
    ///
    /// # Errors
    ///
    /// - See [`Self::rank_items`] for error conditions.
    fn par_rank_items<Id, I, T, A, M>(&self, graph: &Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<usize>, String>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        match self {
            Self::AccumulatedCardinalityRatios(alg) => alg.par_rank_items(graph, tree),
            Self::GraphNeighborhoodSize(alg) => alg.par_rank_items(graph, tree),
            Self::RelativeClusterCardinality(alg) => alg.par_rank_items(graph, tree),
            Self::RelativeComponentCardinality(alg) => alg.par_rank_items(graph, tree),
            Self::RelativeVertexDegree(alg) => alg.par_rank_items(graph, tree),
            Self::StationaryProbabilities(alg) => alg.par_rank_items(graph, tree),
        }
    }

    /// Convert the ranks of items to anomaly scores in the range [0, 1] with higher scores indicating more anomalous items.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn anomaly_scores(ranks: &[usize]) -> Vec<f64> {
        let n = ranks.len();
        ranks.iter().map(|&rank| 1.0 - (rank as f64 / n as f64)).collect()
    }
}

/// An anomaly detection algorithm that can be applied to a Chaoda graph.
///
/// Implementors of this trait should provide the [`Self::raw_anomaly_scores`] method and users should use the [`Self::anomaly_scores`] method to get normalized
/// anomaly scores in the range [0, 1] with higher scores indicating more anomalous items.
trait GraphAlgorithm<Id, I, T, A, M>: NamedAlgorithm
where
    T: DistanceValue,
{
    /// Rank the nodes in the graph based on how anomalous they are, with `1` being the lowest rank and lower ranks indicating more anomalous nodes.
    fn rank_nodes<'a>(&self, graph: &'a Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<(&'a Node<T>, usize)>, String>;

    /// Use the rankings of nodes to assign ranks to the items.
    fn rank_items(&self, graph: &Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<usize>, String> {
        // TODO(Najib): Test this function to see if it correctly handles ties.

        let node_ranks = {
            let mut node_ranks = self.rank_nodes(graph, tree)?;
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
        Ok(ranks)
    }
}

/// Parallel extension of the [`GraphAlgorithm`] trait.
trait ParGraphAlgorithm<Id, I, T, A, M>: GraphAlgorithm<Id, I, T, A, M>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
    /// Parallel version of [`GraphAlgorithm::rank_nodes`], with the default implementation offering no parallelism.
    fn par_rank_nodes<'a>(&self, graph: &'a Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<(&'a Node<T>, usize)>, String> {
        self.rank_nodes(graph, tree)
    }

    /// Parallel version of [`GraphAlgorithm::rank_items`].
    fn par_rank_items(&self, graph: &Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<usize>, String> {
        let node_ranks = {
            let mut node_ranks = self.par_rank_nodes(graph, tree)?;
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
        Ok(ranks)
    }
}
