//! Anomaly detection algorithms using CLAM.

use rayon::prelude::*;

use crate::{DistanceValue, NamedAlgorithm, Tree};

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
    /// Compute anomaly scores for all items from the tree used to create the graph.
    ///
    /// High scores indicate more anomalous nodes, and low scores indicate less anomalous nodes. The scores are normalized to the range [0, 1] using gaussian
    /// error function normalization.
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
    pub fn anomaly_scores<Id, I, T, A, M>(&self, graph: &Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<f64>, String>
    where
        T: DistanceValue,
    {
        match self {
            Self::AccumulatedCardinalityRatios(alg) => alg.anomaly_scores(graph, tree),
            Self::GraphNeighborhoodSize(alg) => alg.anomaly_scores(graph, tree),
            Self::RelativeClusterCardinality(alg) => alg.anomaly_scores(graph, tree),
            Self::RelativeComponentCardinality(alg) => alg.anomaly_scores(graph, tree),
            Self::RelativeVertexDegree(alg) => alg.anomaly_scores(graph, tree),
            Self::StationaryProbabilities(alg) => alg.anomaly_scores(graph, tree),
        }
    }

    /// Parallel version of [`Self::anomaly_scores`].
    ///
    /// # Errors
    ///
    /// See [`Self::anomaly_scores`] for more details on possible errors.
    pub fn par_anomaly_scores<Id, I, T, A, M>(&self, graph: &Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<f64>, String>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
    {
        match self {
            Self::AccumulatedCardinalityRatios(alg) => alg.par_anomaly_scores(graph, tree),
            Self::GraphNeighborhoodSize(alg) => alg.par_anomaly_scores(graph, tree),
            Self::RelativeClusterCardinality(alg) => alg.par_anomaly_scores(graph, tree),
            Self::RelativeComponentCardinality(alg) => alg.par_anomaly_scores(graph, tree),
            Self::RelativeVertexDegree(alg) => alg.par_anomaly_scores(graph, tree),
            Self::StationaryProbabilities(alg) => alg.par_anomaly_scores(graph, tree),
        }
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
    /// Compute anomaly scores for all items from the tree used to create the graph.
    ///
    /// High scores indicate more anomalous nodes, and low scores indicate less anomalous nodes. The scores are not normalized, so they can take any value. They
    /// will be normalized by other methods provided with this trait.
    ///
    /// The returned vector should have the same length as the number of items in the tree and the order of the scores should correspond to the order of the
    /// items in the tree.
    fn raw_anomaly_scores(&self, graph: &Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<f64>, String>;

    /// Compute anomaly scores for all items from the tree used to create the graph, normalized to the range [0, 1] using gaussian error function normalization.
    #[expect(clippy::cast_precision_loss)]
    fn anomaly_scores(&self, graph: &Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<f64>, String> {
        let raw_scores = self.raw_anomaly_scores(graph, tree)?;
        let mean_score = raw_scores.iter().copied().sum::<f64>() / raw_scores.len() as f64;
        let std_dev_score = (raw_scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f64>() / raw_scores.len() as f64).sqrt();
        Ok(raw_scores
            .into_iter()
            // Standardize the scores to have mean 0 and standard deviation 1.
            .map(|s| (s - mean_score) / std_dev_score)
            // Apply the gaussian error function to the standardized scores to the [-1, 1] range.
            .map(libm::erf)
            // Scale the scores to the [0, 1] range.
            .map(|s| f64::midpoint(s, 1.0))
            .collect())
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
    /// Parallel version of [`GraphAlgorithm::raw_anomaly_scores`], with the default implementation offering no parallelism.
    fn par_raw_anomaly_scores(&self, graph: &Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<f64>, String> {
        self.raw_anomaly_scores(graph, tree)
    }

    /// Parallel version of [`GraphAlgorithm::anomaly_scores`].
    #[expect(clippy::cast_precision_loss)]
    fn par_anomaly_scores(&self, graph: &Graph<T>, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<f64>, String> {
        let raw_scores = self.par_raw_anomaly_scores(graph, tree)?;
        let mean_score = raw_scores.par_iter().copied().sum::<f64>() / raw_scores.len() as f64;
        let std_dev_score = (raw_scores.par_iter().map(|s| (s - mean_score).powi(2)).sum::<f64>() / raw_scores.len() as f64).sqrt();
        Ok(raw_scores
            .into_par_iter()
            // Standardize the scores to have mean 0 and standard deviation 1.
            .map(|s| (s - mean_score) / std_dev_score)
            // Apply the gaussian error function to the standardized scores to the [-1, 1] range.
            .map(libm::erf)
            // Scale the scores to the [0, 1] range.
            .map(|s| f64::midpoint(s, 1.0))
            .collect())
    }
}
