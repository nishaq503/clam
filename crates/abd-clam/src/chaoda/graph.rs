//! A `Graph` is a collection of `OddBall`s.

use core::{cmp::Reverse, ops::Index};
use std::collections::BinaryHeap;

use distances::Number;
use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use rayon::prelude::*;

use crate::{Dataset, Instance, Tree};

use super::OddBall;

/// A `Graph` is a collection of `OddBall`s.
///
/// Two `OddBall`s have an edge between them if they have any overlapping volume,
/// i.e. if the distance between their centers is no greater than the sum of their
/// radii.
pub struct Graph<'a, U: Number, C: OddBall<U, N>, const N: usize> {
    /// The collection of `Component`s in the `Graph`.
    components: Vec<Component<'a, U, C, N>>,
    /// Cumulative populations of the `Component`s in the `Graph`.
    populations: Vec<usize>,
}

impl<'a, U: Number, C: OddBall<U, N>, const N: usize> Graph<'a, U, C, N> {
    /// Create a new `Graph` from a `Tree`.
    ///
    /// # Arguments
    ///
    /// * `tree`: The `Tree` to create the `Graph` from.
    /// * `cluster_scorer`: A function that scores a `OddBall`.
    /// * `min_depth`: The minimum depth at which to consider a `OddBall`.
    pub fn from_tree<I: Instance, D: Dataset<I, U>>(
        tree: &'a Tree<I, U, D, C>,
        cluster_scorer: fn(&C) -> f32,
        min_depth: usize,
    ) -> Self {
        // We use `OrderedFloat` to have the `Ord` trait implemented for `f64` so that we can use it in a `BinaryHeap`.
        // We use `Reverse` on `OddBall` so that we can bias towards selecting shallower `OddBall`s.
        // `OddBall`s are selected by highest score and then by shallowest depth.
        let mut candidates = tree
            .root()
            .subtree()
            .into_iter()
            .filter(|c| c.is_leaf() || c.depth() >= min_depth)
            .map(|c| (OrderedFloat(cluster_scorer(c)), Reverse(c)))
            .collect::<BinaryHeap<_>>();

        let mut clusters = vec![];
        while let Some((_, Reverse(c))) = candidates.pop() {
            clusters.push(c);
            // Remove `OddBall`s that are ancestors or descendants of the selected `OddBall`, so as not to have duplicates
            // in the `Graph`.
            candidates.retain(|&(_, Reverse(o))| !c.is_ancestor_of(o) && !c.is_descendant_of(o));
        }

        Self::from_clusters(clusters, tree.data())
    }

    /// Create a new `Graph` from a collection of `OddBall`s.
    pub fn from_clusters<I: Instance, D: Dataset<I, U>>(clusters: Vec<&'a C>, data: &D) -> Self {
        let c = Component::new(clusters, data);
        let [mut c, mut other] = c.partition();
        let mut components = vec![c];
        while !other.is_empty() {
            [c, other] = other.partition();
            components.push(c);
        }
        let populations = components
            .iter()
            .map(|c| c.population)
            .scan(0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect();
        Self {
            components,
            populations,
        }
    }

    /// Iterate over the `OddBall`s in the `Graph`.
    pub fn iter_clusters(&self) -> impl Iterator<Item = &C> {
        self.components.iter().flat_map(Component::iter_clusters)
    }

    /// Iterate over the lists of neighbors of the `OddBall`s in the `Graph`.
    pub fn iter_neighbors(&self) -> impl Iterator<Item = &[(usize, U)]> {
        self.components.iter().flat_map(Component::iter_neighbors)
    }

    /// Get the diameter of the `Graph`.
    pub fn diameter(&mut self) -> usize {
        self.components.iter_mut().map(Component::diameter).max().unwrap_or(0)
    }

    /// Get the neighborhood sizes of all `OddBall`s in the `Graph`.
    pub fn neighborhood_sizes(&mut self) -> Vec<&Vec<usize>> {
        self.components
            .iter_mut()
            .flat_map(Component::neighborhood_sizes)
            .collect()
    }

    /// Get the total number of points in the `Graph`.
    #[must_use]
    pub fn population(&self) -> usize {
        self.populations.last().copied().unwrap_or(0)
    }

    /// Iterate over the `Component`s in the `Graph`.
    pub(crate) fn iter_components(&self) -> impl Iterator<Item = &Component<U, C, N>> {
        self.components.iter()
    }

    /// Compute the stationary probability of each `OddBall` in the `Graph`.
    #[must_use]
    pub fn compute_stationary_probabilities(&self, num_steps: usize) -> Vec<f32> {
        self.components
            .par_iter()
            .flat_map(|c| c.compute_stationary_probabilities(num_steps))
            .collect()
    }
}

/// A `Component` is a single connected subgraph of a `Graph`.
///
/// We break the `Graph` into connected `Component`s because this makes several
/// computations significantly easier to think about and implement.
pub struct Component<'a, U: Number, C: OddBall<U, N>, const N: usize> {
    /// The collection of `OddBall`s in the `Component`.
    clusters: Vec<&'a C>,
    /// The adjacency list of the `Component`. Each `usize` is the index of a `OddBall`
    /// in the `clusters` field and the distance between the two `OddBall`s.
    adjacency_list: Vec<Vec<(usize, U)>>,
    /// The total number of points in the `OddBall`s in the `Component`.
    population: usize,
    /// Eccentricity of each `OddBall` in the `Component`.
    eccentricities: Option<Vec<usize>>,
    /// Diameter of the `Component`.
    diameter: Option<usize>,
    /// neighborhood sizes of each `OddBall` in the `Component` at each step through a BFT.
    neighborhood_sizes: Option<Vec<Vec<usize>>>,
}

impl<'a, U: Number, C: OddBall<U, N>, const N: usize> Component<'a, U, C, N> {
    /// Create a new `Component` from a collection of `OddBall`s.
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

        let population = clusters.iter().map(|c| c.cardinality()).sum();

        Self {
            clusters,
            adjacency_list,
            population,
            eccentricities: None,
            diameter: None,
            neighborhood_sizes: None,
        }
    }

    /// Partition the `Component` into two `Component`s.
    ///
    /// The first component is a connected subgraph of the original `Component`
    /// and the second component is the rest of the original `Component`.
    ///
    /// This method is used when first constructing the `Graph` to find the
    /// connected subgraphs of the `Graph`.
    ///
    /// This method is meant to be used in a loop to find all connected subgraphs
    /// of a `Graph`. It resets the internal members of the `Component` that are
    /// computed lazily, i.e. the eccentricities, diameter, and neighborhood sizes.
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
                    .unwrap_or_else(|| unreachable!("OddBall not found in partitioned component"));
                *j = pos;
            }
        }
        let population = clusters.iter().map(|c| c.cardinality()).sum();
        let other = Self {
            clusters,
            adjacency_list,
            population,
            eccentricities: None,
            diameter: None,
            neighborhood_sizes: None,
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
                    .unwrap_or_else(|| unreachable!("OddBall not found in partitioned component"));
                *j = pos;
            }
        }
        self.clusters = clusters;
        self.adjacency_list = adjacency_list;
        self.population = self.clusters.iter().map(|c| c.cardinality()).sum();
        self.eccentricities = None;
        self.diameter = None;
        self.neighborhood_sizes = None;

        [self, other]
    }

    /// Check if the `Component` has any `OddBall`s.
    fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }

    /// Iterate over the `OddBall`s in the `Component`.
    fn iter_clusters(&self) -> impl Iterator<Item = &C> {
        self.clusters.iter().copied()
    }

    /// Iterate over the lists of neighbors of the `OddBall`s in the `Component`.
    fn iter_neighbors(&self) -> impl Iterator<Item = &[(usize, U)]> {
        self.adjacency_list.iter().map(Vec::as_slice)
    }

    /// Get the number of `OddBall`s in the `Component`.
    pub fn cardinality(&self) -> usize {
        self.clusters.len()
    }

    /// Get the total number of points in the `Component`.
    pub const fn population(&self) -> usize {
        self.population
    }

    /// Get the diameter of the `Component`.
    pub fn diameter(&mut self) -> usize {
        if self.diameter.is_none() {
            if self.eccentricities.is_none() {
                self.compute_eccentricities();
            }
            let ecc = self
                .eccentricities
                .as_ref()
                .unwrap_or_else(|| unreachable!("We just computed the eccentricities"));
            self.diameter = Some(ecc.iter().copied().max().unwrap_or(0));
        }
        self.diameter
            .unwrap_or_else(|| unreachable!("We just computed the diameter"))
    }

    /// Compute the eccentricity of each `OddBall` in the `Component`.
    pub fn compute_eccentricities(&mut self) {
        self.eccentricities = Some(self.neighborhood_sizes().iter().map(Vec::len).collect());
    }

    /// Get the neighborhood sizes of all `OddBall`s in the `Component`.
    pub fn neighborhood_sizes(&mut self) -> &[Vec<usize>] {
        if self.neighborhood_sizes.is_none() {
            self.neighborhood_sizes = Some(
                (0..self.cardinality())
                    .map(|i| self.compute_neighborhood_sizes(i))
                    .collect(),
            );
        }
        self.neighborhood_sizes
            .as_ref()
            .unwrap_or_else(|| unreachable!("We just computed the neighborhood sizes"))
    }

    /// Get the cumulative number of neighbors encountered after each step through a BFT.
    fn compute_neighborhood_sizes(&self, i: usize) -> Vec<usize> {
        let mut visited = vec![false; self.cardinality()];
        let mut neighborhood_sizes = Vec::new();
        let mut stack = vec![i];
        while let Some(i) = stack.pop() {
            if visited[i] {
                continue;
            }
            visited[i] = true;
            let new_neighbors = self.adjacency_list[i]
                .iter()
                .filter(|(j, _)| !visited[*j])
                .collect::<Vec<_>>();
            neighborhood_sizes.push(new_neighbors.len());
            stack.extend(new_neighbors.iter().map(|(j, _)| *j));
        }

        neighborhood_sizes
            .iter()
            .scan(0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect()
    }

    /// Compute the stationary probability of each `OddBall` in the `Component`.
    pub fn compute_stationary_probabilities(&self, num_steps: usize) -> Vec<f32> {
        let mut transition_matrix = vec![0_f32; self.cardinality() * self.cardinality()];
        for (i, neighbors) in self.adjacency_list.iter().enumerate() {
            for &(j, d) in neighbors {
                transition_matrix[i * self.cardinality() + j] = 1.0 / d.as_f32();
            }
        }
        // Convert the transition matrix to an Array2
        let mut transition_matrix = Array2::from_shape_vec((self.cardinality(), self.cardinality()), transition_matrix)
            .unwrap_or_else(|e| unreachable!("We created a square Transition matrix: {e}"));

        // Normalize the transition matrix so that each row sums to 1
        for i in 0..self.cardinality() {
            let row_sum = transition_matrix.row(i).sum();
            transition_matrix.row_mut(i).mapv_inplace(|x| x / row_sum);
        }

        // Compute the stationary probabilities by squaring the transition matrix `num_steps` times
        for _ in 0..num_steps {
            transition_matrix = transition_matrix.dot(&transition_matrix);
        }

        // Compute the stationary probabilities by summing the rows of the transition matrix
        transition_matrix.sum_axis(Axis(1)).to_vec()
    }
}

impl<'a, U: Number, C: OddBall<U, N>, const N: usize> Index<usize> for Component<'a, U, C, N> {
    type Output = C;

    fn index(&self, index: usize) -> &Self::Output {
        self.clusters[index]
    }
}
