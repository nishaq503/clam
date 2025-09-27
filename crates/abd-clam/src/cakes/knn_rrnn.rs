//! K-Nearest Neighbor (KNN) search using the Repeated Radius Nearest Neighbor (RRNN) algorithm.

use rayon::prelude::*;

use crate::{utils::SizedHeap, Ball, DistanceValue};

use super::{
    rnn_chess::{par_tree_search, tree_search},
    ParSearch, Search,
};

/// K-Nearest Neighbor (KNN) search using the Repeated Radius Nearest Neighbor (RRNN) algorithm.
pub struct KnnRrnn(pub usize);

impl<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A> Search<Id, I, T, M, A> for KnnRrnn {
    fn search<'a>(&self, root: &'a Ball<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        profi::prof!("KnnRrnn::search");

        if self.0 > root.cardinality() {
            // If k is greater than the number of points in the tree, return all
            // items with their distances.
            return root.distances_to_all_items(query, metric);
        }

        // Estimate an initial radius to cover k points.
        let mut radius = radius_for_k(root, self.0);

        // Perform the initial tree search.
        let (mut centers, mut subsumed, mut straddlers) = {
            profi::prof!("KnnRrnn::search::tree_search");
            tree_search(
                root,
                metric,
                query,
                T::from_f64(radius)
                    .unwrap_or_else(|| unreachable!("f64 to {} conversion failed", std::any::type_name::<T>())),
            )
        };

        // Count the number of confirmed hits.
        let mut num_confirmed = count_hits(&centers, &subsumed);
        while num_confirmed < self.0 {
            profi::prof!("KnnRrnn::search::while_loop");

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
            (centers, subsumed, straddlers) = {
                profi::prof!("KnnRrnn::search::tree_search");
                tree_search(
                    root,
                    metric,
                    query,
                    T::from_f64(radius)
                        .unwrap_or_else(|| unreachable!("f64 to {} conversion failed", std::any::type_name::<T>())),
                )
            };
            // Recount the number of confirmed hits.
            num_confirmed = count_hits(&centers, &subsumed);
        }

        // We now have at least k confirmed hits; collect them.
        let mut heap = SizedHeap::<(&Id, &I), T>::new(Some(self.0));
        {
            profi::prof!("KnnRrnn::search::leaf_search");
            heap.extend(centers.into_iter().map(|(id, item, d)| ((id, item), d)));

            for ball in subsumed.into_iter().chain(straddlers) {
                if ball.is_singleton() {
                    let d = metric(query, ball.center());
                    heap.extend(ball.subtree_items().iter().map(|(id, item)| ((id, item), d)));
                } else {
                    heap.extend(
                        ball.subtree_items()
                            .iter()
                            .map(|(id, item)| ((id, item), metric(query, item))),
                    );
                }
            }
        }

        heap.items().map(|((id, item), d)| (id, item, d)).collect()
    }
}

impl<
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
        A: Send + Sync,
    > ParSearch<Id, I, T, M, A> for KnnRrnn
{
    fn par_search<'a>(&self, root: &'a Ball<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        profi::prof!("KnnRrnn::search");

        if self.0 > root.cardinality() {
            // If k is greater than the number of points in the tree, return all
            // items with their distances.
            return root.par_distances_to_all_items(query, metric);
        }

        // Estimate an initial radius to cover k points.
        let mut radius = radius_for_k(root, self.0);

        // Perform the initial tree search.
        let (mut centers, mut subsumed, mut straddlers) = {
            profi::prof!("KnnRrnn::search::tree_search");
            par_tree_search(
                root,
                metric,
                query,
                T::from_f64(radius)
                    .unwrap_or_else(|| unreachable!("f64 to {} conversion failed", std::any::type_name::<T>())),
            )
        };

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
            (centers, subsumed, straddlers) = {
                profi::prof!("KnnRrnn::search::tree_search");
                par_tree_search(
                    root,
                    metric,
                    query,
                    T::from_f64(radius)
                        .unwrap_or_else(|| unreachable!("f64 to {} conversion failed", std::any::type_name::<T>())),
                )
            };
            // Recount the number of confirmed hits.
            num_confirmed = count_hits(&centers, &subsumed);
        }

        // We now have at least k confirmed hits; collect them.
        let mut heap = SizedHeap::<(&Id, &I), T>::new(Some(self.0));
        {
            profi::prof!("KnnRrnn::search::leaf_search");

            heap.extend(centers.into_iter().map(|(id, item, d)| ((id, item), d)));

            for ball in subsumed.into_iter().chain(straddlers) {
                if ball.is_singleton() {
                    let d = metric(query, ball.center());
                    heap.extend(ball.subtree_items().iter().map(|(id, item)| ((id, item), d)));
                } else {
                    heap.extend(
                        ball.subtree_items()
                            .par_iter()
                            .map(|(id, item)| ((id, item), metric(query, item)))
                            .collect::<Vec<_>>(),
                    );
                }
            }
        }

        heap.items().map(|((id, item), d)| (id, item, d)).collect()
    }
}

/// Computes the radius needed to cover k points from the cluster center.
#[expect(clippy::cast_precision_loss)]
fn radius_for_k<I, Id, T: DistanceValue, A>(ball: &Ball<I, Id, T, A>, k: usize) -> f64 {
    let r = ball
        .radius()
        .to_f64()
        .unwrap_or_else(|| unreachable!("Radius of type {} to f64 conversion failed", std::any::type_name::<T>()));
    if ball.cardinality() == k {
        r
    } else {
        r * (k as f64 / ball.cardinality() as f64).powf(ball.lfd().recip())
    }
}

/// Counts the total number of hits from confirmed centers and subsumed balls.
fn count_hits<Id, I, T: DistanceValue, A>(centers: &[(&Id, &I, T)], subsumed: &[&Ball<Id, I, T, A>]) -> usize {
    centers.len()
        + subsumed
            .iter()
            .map(|b| b.cardinality() - 1) // -1 because we already have the centers
            .sum::<usize>()
}

/// Calculate a multiplier for the radius using the LFDs of the clusters.
#[expect(clippy::cast_precision_loss)]
fn lfd_multiplier<Id, I, T: DistanceValue, A>(
    centers: &[(&Id, &I, T)],
    subsumed: &[&Ball<Id, I, T, A>],
    straddlers: &[&Ball<Id, I, T, A>],
    k: usize,
    num_confirmed: usize,
) -> f64 {
    let radial_distances = centers.iter().map(|&(_, _, d)| d).collect::<Vec<_>>();
    let radius = radial_distances
        .iter()
        .max_by_key(|&&d| crate::utils::MaxItem((), d))
        .map_or_else(T::zero, |&d| d);
    let lfd_recip_sum_init = crate::core::lfd_estimate(&radial_distances, radius).recip();

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
