//! K-nearest neighbors (KNN) search using the Greedy Branch algorithm.

use std::cmp::Reverse;

use crate::{
    DistanceValue, Tree,
    cakes::{RnnChess, Search, d_max, d_min},
    utils::SizedHeap,
};

/// K-nearest neighbors (KNN) search using the Greedy Branch algorithm.
///
/// The field is the number of nearest neighbors to find (k).
pub struct KnnBranch(pub usize);

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for KnnBranch
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn name(&self) -> String {
        format!("KnnBranch(k={})", self.0)
    }

    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        if self.0 > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all
            // items with their distances.
            return tree.distances_to_items_in_cluster(query, tree.root());
        }

        let mut candidate_radii = SizedHeap::<usize, Reverse<T>>::new(None);

        let d = tree.distance_to_center(query, tree.root());
        candidate_radii.push((1, Reverse(d_min(tree.root(), d))));
        candidate_radii.push((tree.cardinality().half() + 1, Reverse(d)));
        candidate_radii.push((tree.cardinality(), Reverse(d_max(tree.root(), d))));

        let mut latest = tree.root();
        while !latest.is_leaf() {
            let (child, d) = latest
                .children()
                .unwrap_or_else(|| unreachable!("We checked is_leaf above"))
                .iter()
                .map(|c| (c, tree.distance_to_center(query, c)))
                .min_by_key(|&(_, d)| crate::utils::MinItem((), d))
                .unwrap_or_else(|| unreachable!("We checked is_leaf above"));

            candidate_radii.push((1, Reverse(d_min(child, d))));
            candidate_radii.push((child.cardinality().half() + 1, Reverse(d)));
            candidate_radii.push((child.cardinality(), Reverse(d_max(child, d))));

            latest = child;
        }
        // Remove the candidate radii that are zero.
        let mut expected_hits = 0;
        while candidate_radii.peek().is_some_and(|(&e, &Reverse(d))| {
            expected_hits += e;
            d.is_zero() && expected_hits <= self.0
        }) {
            candidate_radii.pop();
        }

        // Now, try CHESS with increasing candidate radii until we have k hits.
        let mut hits = Vec::new();
        while let Some((_, Reverse(d))) = candidate_radii.pop() {
            hits = RnnChess(d).search(tree, query);
            if hits.len() >= self.0 {
                hits.sort_by_key(|&(_, d)| crate::utils::MinItem((), d));
                hits.truncate(self.0);
                break;
            }
        }
        hits
    }
}
