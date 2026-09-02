//! A `Node` is more anomalous if it can reach fewer other nodes in the graph within the same number of steps as compared to other nodes in the graph.

use rayon::prelude::*;

use crate::{DistanceValue, NamedAlgorithm, Tree, chaoda::Node};

use super::{AnomalyFeatures, Graph, GraphAlgorithm};

/// A `Node` is more anomalous if it can reach fewer other nodes in the graph within the same number of steps as compared to other nodes in the graph.
#[derive(Debug, Clone)]
#[must_use]
pub struct GraphNeighborhoodSize;

impl_named_algorithm_for_unit_struct!(GraphNeighborhoodSize, "graph-neighborhood-size", r"^graph-neighborhood-size$");

impl<Id, I, T, A, M> GraphAlgorithm<Id, I, T, A, M> for GraphNeighborhoodSize
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
        neighborhood_sizes
            .into_iter()
            .flat_map(|(node, sizes)| {
                sizes
                    .into_iter()
                    .take(max_steps)
                    .scan(0, |acc, x| {
                        *acc += x;
                        Some(*acc)
                    })
                    // A larger cumulative neighborhood size indicates that the node is more central, and thus less anomalous.
                    .map(move |score| (node, score))
            })
            .collect::<Vec<_>>()
    }

    fn par_rank_nodes<'a>(&self, graph: &'a Graph<T>, _: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> Vec<(&'a Node<T>, usize)>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
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
        let diameter = neighborhood_sizes.iter().map(|(_, sizes)| sizes.len()).max().unwrap_or(0);

        // We will consider a node's neighborhood up to a quarter of the graph's diameter.
        let max_steps = diameter / 4;

        // For each node, we will calculate the cumulative neighborhood size up to `max_steps`.
        neighborhood_sizes
            .into_par_iter()
            .flat_map(|(node, sizes)| {
                sizes
                    .into_iter()
                    .take(max_steps)
                    .scan(0, |acc, x| {
                        *acc += x;
                        Some(*acc)
                    })
                    // A larger cumulative neighborhood size indicates that the node is more central, and thus less anomalous.
                    .map(move |score| (node, score))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }
}
