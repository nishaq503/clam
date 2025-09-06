//! K-Nearest Neighbor (KNN) search using the Repeated Radius Nearest Neighbor (RRNN) algorithm.

use crate::{
    Cluster, DistanceValue, Tree,
    cakes::{ParSearch, Search},
    tree::lfd_estimate,
    utils::{MaxItem, SizedHeap},
};

use super::rnn_chess::tree_search;

/// K-Nearest Neighbor (KNN) search using the Repeated Radius Nearest Neighbor (RRNN) algorithm.
///
/// The field is the number of nearest neighbors to find (k).
pub struct KnnRrnn(pub usize);

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnRrnn {
    fn name(&self) -> String {
        format!("KnnRrnn(k={})", self.0)
    }

    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        if self.0 > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all
            // items with their distances.
            return tree.distances_to_items_in_cluster(query, &tree.root);
        }

        // Estimate an initial radius to cover k points.
        let mut radius = radius_for_k(&tree.root, self.0);

        // Perform the initial tree search.
        let (mut centers, mut subsumed, mut straddlers) = tree_search(
            tree,
            &tree.root,
            query,
            T::from_f64(radius)
                .unwrap_or_else(|| unreachable!("f64 to {} conversion failed", std::any::type_name::<T>())),
        );

        // Count the number of confirmed hits.
        let mut num_confirmed = count_hits(&centers, &subsumed);
        while num_confirmed < self.0 {
            // While we don't have enough hits...
            let multiplier = if num_confirmed == 0 {
                // If no hits, double the radius.
                2.0
            } else {
                // Otherwise, calculate a multiplier based on LFDs.
                lfd_multiplier(&centers, &subsumed, &straddlers, self.0, num_confirmed)
            };

            // Increase the radius and repeat the search.
            radius *= multiplier;
            (centers, subsumed, straddlers) = tree_search(
                tree,
                &tree.root,
                query,
                T::from_f64(radius)
                    .unwrap_or_else(|| unreachable!("f64 to {} conversion failed", std::any::type_name::<T>())),
            );
            // Recount the number of confirmed hits.
            num_confirmed = count_hits(&centers, &subsumed);
        }

        // We now have at least k confirmed hits; collect them.
        let mut heap = SizedHeap::<usize, T>::new(Some(self.0));
        heap.extend(centers);

        for cluster in subsumed.into_iter().chain(straddlers) {
            if cluster.is_singleton() {
                // TODO(Najib): Figure out how to get this distance from the heap
                let d = tree.distance_to_center(query, cluster);
                heap.extend(cluster.subtree_indices().map(|i| (i, d)));
            } else {
                heap.extend(tree.distances_to_items_in_subtree(query, cluster));
            }
        }

        heap.take_items().collect()
    }
}

impl<Id, I, T, A, M> ParSearch<Id, I, T, A, M> for KnnRrnn
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        // For now, just call the single-threaded search.
        self.search(tree, query)
    }
}

/// Computes the radius needed to cover k points from the cluster center.
#[expect(clippy::cast_precision_loss)]
fn radius_for_k<T: DistanceValue, A>(cluster: &Cluster<T, A>, k: usize) -> f64 {
    let r = cluster
        .radius()
        .to_f64()
        .unwrap_or_else(|| unreachable!("Radius of type {} to f64 conversion failed", std::any::type_name::<T>()));
    if cluster.cardinality() == k {
        r
    } else {
        r * (k as f64 / cluster.cardinality() as f64).powf(cluster.lfd().recip())
    }
}

/// Counts the total number of hits from confirmed centers and subsumed clusters.
fn count_hits<T: DistanceValue, A>(centers: &[(usize, T)], subsumed: &[&Cluster<T, A>]) -> usize {
    centers.len()
        + subsumed
            .iter()
            .map(|b| b.cardinality() - 1) // -1 because we already have the centers
            .sum::<usize>()
}

/// Calculate a multiplier for the radius using the LFDs of the clusters.
#[expect(clippy::cast_precision_loss)]
fn lfd_multiplier<T: DistanceValue, A>(
    centers: &[(usize, T)],
    subsumed: &[&Cluster<T, A>],
    straddlers: &[&Cluster<T, A>],
    k: usize,
    num_confirmed: usize,
) -> f64 {
    let radial_distances = centers.iter().map(|&(_, d)| d).collect::<Vec<_>>();
    let radius = radial_distances
        .iter()
        .max_by_key(|&&d| MaxItem((), d))
        .map_or_else(T::zero, |&d| d);
    let lfd_recip_sum_init = lfd_estimate(&radial_distances, radius).recip();

    let lfd_recip_sum = lfd_recip_sum_init
        + subsumed
            .iter()
            .chain(straddlers.iter())
            .map(|b| b.lfd().recip())
            .sum::<f64>();

    let n_lfd_samples = subsumed.len() + straddlers.len() + 1; // +1 for the `centers` list
    let lfd_harmonic_mean_inv = lfd_recip_sum / n_lfd_samples as f64;
    (k as f64 / num_confirmed as f64)
        .powf(lfd_harmonic_mean_inv)
        .next_up()
        .clamp(1_f64.next_up(), 2.0)
}
