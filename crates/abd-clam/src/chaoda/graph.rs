//! A `Graph` is a collection of `Cluster`s.

#![allow(dead_code, unused_variables)]

use core::{cmp::Reverse, ops::Index};
use std::collections::BinaryHeap;

use distances::Number;
use ordered_float::OrderedFloat;
use rayon::prelude::*;

use crate::{Cluster, Dataset, Instance, Tree};

/// A `Graph` is a collection of `Cluster`s.
///
/// Two `Cluster`s have an edge between them if they have any overlapping volume,
/// i.e. if the distance between their centers is no greater than the sum of their
/// radii.
pub struct Graph<'a, U: Number, C: Cluster<U>> {
    /// The collection of `Component`s in the `Graph`.
    components: Vec<Component<'a, U, C>>,
}

impl<'a, U: Number, C: Cluster<U>> Graph<'a, U, C> {
    /// Create a new `Graph` from a `Tree`.
    ///
    /// # Arguments
    ///
    /// * `tree`: The `Tree` to create the `Graph` from.
    /// * `cluster_scorer`: A function that scores a `Cluster`.
    /// * `min_depth`: The minimum depth at which to consider a `Cluster`.
    pub fn from_tree<I: Instance, D: Dataset<I, U>>(
        tree: &'a Tree<I, U, D, C>,
        cluster_scorer: fn(&C) -> f64,
        min_depth: usize,
    ) -> Self {
        let root = tree.root();
        let data = tree.data();

        // We use `OrderedFloat` to have the `Ord` trait implemented for `f64` so that we can use it in a `BinaryHeap`.
        // We use `Reverse` on `Cluster` so that we can bias towards selecting shallower `Cluster`s.
        // `Cluster`s are selected by highest score and then by shallowest depth.
        let mut candidates = tree
            .root()
            .subtree()
            .into_iter()
            .filter(|c| c.depth() >= min_depth)
            .map(|c| (OrderedFloat(cluster_scorer(c)), Reverse(c)))
            .collect::<BinaryHeap<_>>();

        let mut clusters = vec![];
        while let Some((_, Reverse(c))) = candidates.pop() {
            clusters.push(c);
            // Remove `Cluster`s that are ancestors or descendants of the selected `Cluster`, so as not to have duplicates
            // in the `Graph`.
            candidates.retain(|&(_, Reverse(o))| !c.is_ancestor_of(o) && !c.is_descendant_of(o));
        }

        Self::from_clusters(clusters, data)
    }

    /// Create a new `Graph` from a collection of `Cluster`s.
    pub fn from_clusters<I: Instance, D: Dataset<I, U>>(clusters: Vec<&'a C>, data: &D) -> Self {
        let c = Component::new(clusters, data);
        let [mut c, mut other] = c.partition();
        let mut components = vec![c];
        while !other.is_empty() {
            [c, other] = other.partition();
            components.push(c);
        }
        Self { components }
    }
}

/// A `Component` is a single connected subgraph of a `Graph`.
///
/// We break the `Graph` into connected `Component`s because this makes several
/// computations significantly easier to think about and implement.
struct Component<'a, U: Number, C: Cluster<U>> {
    /// The collection of `Cluster`s in the `Component`.
    clusters: Vec<&'a C>,
    /// The adjacency list of the `Component`. Each `usize` is the index of a `Cluster`
    /// in the `clusters` field and the distance between the two `Cluster`s.
    adjacency_list: Vec<Vec<(usize, U)>>,
}

impl<'a, U: Number, C: Cluster<U>> Component<'a, U, C> {
    /// Create a new `Component` from a collection of `Cluster`s.
    fn new<I: Instance, D: Dataset<I, U>>(clusters: Vec<&'a C>, data: &D) -> Self {
        let adjacency_list = clusters
            .par_iter()
            .enumerate()
            .map(|(i, c1)| {
                clusters
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| i != j)
                    .filter_map(|(j, c2)| {
                        let (r1, r2) = (c1.radius(), c2.radius());
                        let d = c1.distance_to_other(data, c2);
                        if d <= r1 + r2 {
                            Some((j, d))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        Self {
            clusters,
            adjacency_list,
        }
    }

    /// Partition the `Component` into two `Component`s.
    ///
    /// The first component is a connected subgraph of the original `Component`
    /// and the second component is the rest of the original `Component`.
    ///
    /// This method is used when first constructing the `Graph` to find the
    /// connected subgraphs of the `Graph`.
    fn partition(mut self) -> [Self; 2] {
        // Perform a traversal of the adjacency list to find a connected subgraph.
        let mut visited = vec![false; self.clusters.len()];
        let mut stack = vec![0];
        while let Some(i) = stack.pop() {
            if visited[i] {
                continue;
            }
            visited[i] = true;
            for &(j, _) in &self.adjacency_list[i] {
                stack.push(j);
            }
        }
        let (c1, c2) = visited
            .iter()
            .zip(self.clusters.iter().copied())
            .partition::<Vec<_>, _>(|&(&v, _)| v);
        let (a1, a2) = visited
            .iter()
            .zip(self.adjacency_list)
            .partition::<Vec<_>, _>(|&(&v, _)| v);

        // Build a component from the clusters that were not visited in the traversal.
        let clusters = c2.into_iter().map(|(_, c)| c).collect::<Vec<_>>();
        let mut adjacency_list = a2.into_iter().map(|(_, a)| a).collect::<Vec<_>>();
        // Remap indices in adjacency list
        for a in &mut adjacency_list {
            for (j, _) in a {
                let old_c = self.clusters[*j];
                let pos = clusters
                    .iter()
                    .position(|c| c.name() == old_c.name())
                    .unwrap_or_else(|| unreachable!("Cluster not found in partitioned component"));
                *j = pos;
            }
        }
        let other = Self {
            clusters,
            adjacency_list,
        };

        // Set the current component to the visited clusters.
        let clusters = c1.into_iter().map(|(_, c)| c).collect::<Vec<_>>();
        let mut adjacency_list = a1.into_iter().map(|(_, a)| a).collect::<Vec<_>>();
        // Remap indices in adjacency list
        for a in &mut adjacency_list {
            for (j, _) in a {
                let old_c = self.clusters[*j];
                let pos = clusters
                    .iter()
                    .position(|c| c.name() == old_c.name())
                    .unwrap_or_else(|| unreachable!("Cluster not found in partitioned component"));
                *j = pos;
            }
        }
        self.clusters = clusters;
        self.adjacency_list = adjacency_list;

        [self, other]
    }

    /// Check if the `Component` has any `Cluster`s.
    fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }
}

impl<'a, U: Number, C: Cluster<U>> Index<usize> for Component<'a, U, C> {
    type Output = C;

    fn index(&self, index: usize) -> &Self::Output {
        self.clusters[index]
    }
}
