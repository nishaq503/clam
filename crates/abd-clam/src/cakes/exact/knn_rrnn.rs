//! K-Nearest Neighbor (KNN) search using the Repeated Radius Nearest Neighbor (RRNN) algorithm.

use crate::{
    cakes::Search,
    tree::lfd_estimate,
    utils::{MaxItem, SizedHeap},
    Cluster, DistanceValue, Tree,
};

use super::rnn_chess::tree_search;

/// K-Nearest Neighbor (KNN) search using the Repeated Radius Nearest Neighbor (RRNN) algorithm.
///
/// The field is the number of nearest neighbors to find (k).
pub struct KnnRrnn(pub usize);

impl std::fmt::Display for KnnRrnn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KnnRrnn(k={})", self.0)
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnRrnn {
    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();
        let metric = tree.metric();
        let items = tree.items();

        if self.0 > items.len() {
            // If k is greater than the number of points in the tree, return all
            // items with their distances.
            return items
                .iter()
                .enumerate()
                .map(|(i, (_, item))| (i, metric(query, item)))
                .collect();
        }

        // Estimate an initial radius to cover k points.
        let mut radius = radius_for_k(root, self.0);

        // Perform the initial tree search.
        let (mut centers, mut subsumed, mut straddlers) = tree_search(
            root,
            metric,
            items,
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
                root,
                metric,
                items,
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
                let d = metric(query, &items[cluster.center_index()].1);
                heap.extend(cluster.subtree_indices().map(|i| (i, d)));
            } else {
                heap.extend(cluster.subtree_indices().map(|i| (i, metric(query, &items[i].1))));
            }
        }

        heap.take_items().collect()
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
