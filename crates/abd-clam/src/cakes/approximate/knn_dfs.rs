//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::cmp::Reverse;

use crate::{
    cakes::{d_max, d_min, leaf_into_hits, pop_till_leaf, BatchedSearch, Search},
    utils::SizedHeap,
    Cluster, DistanceValue,
};

/// K-Nearest Neighbor (KNN) search using the Depth-First Sieve algorithm.
///
/// The fields are:
///   1. `k`: The number of nearest neighbors to find.
///   2. `max_leaves`: The maximum number of leaf nodes to visit (`usize::MAX` for no limit).
///   3. `max_dist_comps`: The maximum number of distance computations to perform (`usize::MAX` for no limit).
///
/// If both `max_leaves` and `max_dist_comps` are set to `usize::MAX`, the search is exact and will have the same asymptotic behavior as the
/// [`exact variant`](crate::cakes::KnnDfs) of this algorithm.
pub struct KnnDfs(pub usize, pub usize, pub usize);

impl std::fmt::Display for KnnDfs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.1 == usize::MAX && self.2 == usize::MAX {
            write!(f, "KnnDfs(k={})", self.0)
        } else if self.2 == usize::MAX {
            write!(f, "ApproxKnnDfs(k={},leaves<{})", self.0, self.1)
        } else if self.1 == usize::MAX {
            write!(f, "ApproxKnnDfs(k={},dist_comps<{})", self.0, self.2)
        } else {
            write!(f, "ApproxKnnDfs(k={},leaves<{},dist_comps<{})", self.0, self.1, self.2)
        }
    }
}

impl<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A> Search<Id, I, T, M, A> for KnnDfs {
    fn search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        if self.0 > root.cardinality() {
            // If k is greater than the number of points in the tree, return all
            // items with their distances.
            return root.distances_to_all_items(query, metric);
        }
        // let tol = 0.01; // Tolerance for hit improvement.

        let mut candidates = SizedHeap::<&'a Cluster<Id, I, T, A>, Reverse<(T, T, T)>>::new(None);
        let mut hits = SizedHeap::<(&'a Id, &'a I), T>::new(Some(self.0));

        let d = metric(query, root.center());
        hits.push(((root.center_id(), root.center()), d));
        candidates.push((root, Reverse((d_min(root, d), d_max(root, d), d))));

        let mut leaves_visited = 0;
        let mut distance_computations = 1;

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf, d, n) = pop_till_leaf(query, metric, &mut candidates, &mut hits);
            leaves_visited += 1;
            distance_computations += n;

            // Process the leaf and update hits.
            distance_computations += leaf_into_hits(query, metric, &mut hits, leaf, d);

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates
                .peek()
                .map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);

            if hits.is_full()
                && [
                    max_h < min_c,                   // All remaining candidates are too far to improve hits.
                    leaves_visited >= self.1,        // Stop after visiting this many leaves.
                    distance_computations >= self.2, // Stop after this many distance computations.
                ]
                .iter()
                .any(|&x| x)
            {
                // The closest candidate cannot improve our hits, so we can stop.
                break;
            }
        }

        hits.take_items().map(|((id, item), d)| (id, item, d)).collect()
    }
}

impl<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A> BatchedSearch<Id, I, T, M, A> for KnnDfs {}
