//! A `Node` is more anomalous if it can reach fewer other nodes in the graph within the same number of steps as compared to other nodes in the graph.

use rayon::prelude::*;

use crate::{DistanceValue, NamedAlgorithm, Tree};

use super::{AnomalyFeatures, Graph, GraphAlgorithm, ParGraphAlgorithm};

/// A `Node` is more anomalous if it can reach fewer other nodes in the graph within the same number of steps as compared to other nodes in the graph.
#[derive(Debug, Clone)]
#[must_use]
pub struct GraphNeighborhoodSize;

impl_named_algorithm_for_unit_struct!(GraphNeighborhoodSize, "graph-neighborhood-size", r"^graph-neighborhood-size$");

impl<Id, I, T, A, M> GraphAlgorithm<Id, I, T, A, M> for GraphNeighborhoodSize
where
    T: DistanceValue,
{
    #[expect(clippy::cast_precision_loss)]
    fn raw_anomaly_scores(&self, graph: &Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<f64>, String> {
        let neighborhood_sizes = graph
            .iter_components()
            .flat_map(|component| {
                component.iter_nodes().map(move |node| {
                    let neighborhood_sizes = component.reachable_nodes_by_steps(node).into_iter().map(|v| v.len()).collect::<Vec<_>>();
                    (node, neighborhood_sizes)
                })
            })
            .collect::<Vec<_>>();

        // The diameter of the graph is the maximum eccentricity of any node, which is the maximum length of all shortest paths between any two nodes.
        let diameter = neighborhood_sizes.iter().map(|(_, sizes)| sizes.len()).max().unwrap_or(0);

        // We will consider a node's neighborhood up to a quarter of the graph's diameter.
        let max_steps = diameter / 4;

        // For each node, we will calculate the cumulative neighborhood size up to `max_steps`.
        let cumulative_neighborhood_sizes = neighborhood_sizes.into_iter().flat_map(|(node, sizes)| {
            sizes
                .into_iter()
                .take(max_steps)
                .scan(0, |acc, x| {
                    *acc += x;
                    Some(*acc)
                })
                // A larger cumulative neighborhood size indicates that the node is more central, and thus less anomalous.
                .map(move |score| (node, -(score as f64)))
        });

        // All items in a node inherit the same anomaly score as the node itself.
        let mut scores = cumulative_neighborhood_sizes
            .flat_map(|(node, score)| node.iter_items().map(move |item| (item, score)))
            .collect::<Vec<_>>();

        // Sort the scores by item index to ensure they are in the same order as the input data.
        scores.sort_by_key(|(i, _)| *i);
        Ok(scores.into_iter().map(|(_, score)| score).collect())
    }
}

impl<Id, I, T, A, M> ParGraphAlgorithm<Id, I, T, A, M> for GraphNeighborhoodSize
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
    #[expect(clippy::cast_precision_loss)]
    fn par_raw_anomaly_scores(&self, graph: &Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Result<Vec<f64>, String> {
        let neighborhood_sizes = graph
            .par_iter_components()
            .flat_map(|component| {
                component.par_iter_nodes().map(move |node| {
                    let neighborhood_sizes = component.reachable_nodes_by_steps(node).into_iter().map(|v| v.len()).collect::<Vec<_>>();
                    (node, neighborhood_sizes)
                })
            })
            .collect::<Vec<_>>();

        // The diameter of the graph is the maximum eccentricity of any node, which is the maximum length of all shortest paths between any two nodes.
        let diameter = neighborhood_sizes.par_iter().map(|(_, sizes)| sizes.len()).max().unwrap_or(0);

        // We will consider a node's neighborhood up to a quarter of the graph's diameter.
        let max_steps = diameter / 4;

        // For each node, we will calculate the cumulative neighborhood size up to `max_steps`.
        let cumulative_neighborhood_sizes = neighborhood_sizes.into_iter().flat_map(|(node, sizes)| {
            sizes
                .into_iter()
                .take(max_steps)
                .scan(0, |acc, x| {
                    *acc += x;
                    Some(*acc)
                })
                // A larger cumulative neighborhood size indicates that the node is more central, and thus less anomalous.
                .map(move |score| (node, -(score as f64)))
        });

        // All items in a node inherit the same anomaly score as the node itself.
        let mut scores = cumulative_neighborhood_sizes
            .flat_map(|(node, score)| node.iter_items().map(move |item| (item, score)))
            .collect::<Vec<_>>();

        // Sort the scores by item index to ensure they are in the same order as the input data.
        scores.sort_by_key(|(i, _)| *i);
        Ok(scores.into_iter().map(|(_, score)| score).collect())
    }
}
