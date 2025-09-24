//! K-Nearest Neighbors (KNN) search using the Breadth-First Sieve algorithm.

use crate::{utils::SizedHeap, Ball, DistanceValue};

use super::{ParSearch, Search};

/// K-Nearest Neighbor (KNN) search using the Breadth-First Sieve algorithm.
pub struct KnnBfs(pub usize);

impl<I, T: DistanceValue> Search<I, T> for KnnBfs {
    fn search<'a, M: Fn(&I, &I) -> T>(&self, root: &'a Ball<I, T>, metric: &M, query: &I) -> Vec<(&'a I, T)> {
        if self.0 > root.cardinality() {
            // If k is greater than the number of points in the tree, return all
            // items with their distances.
            return root.all_items().into_iter().map(|p| (p, metric(query, p))).collect();
        }

        let mut candidates = Vec::new();
        let mut hits = SizedHeap::<&I, T>::new(Some(self.0));

        let d = metric(query, root.center());
        hits.push((root.center(), d));
        candidates.push((root, d_max(root, d)));

        while !candidates.is_empty() {
            candidates =
                filter_candidates(candidates, self.0)
                    .into_iter()
                    .fold(Vec::new(), |mut acc_candidates, (ball, d)| {
                        if (
                        acc_candidates.len() <= self.0  // We still need more points to satisfy k, AND
                        && (ball.cardinality() < (self.0 - acc_candidates.len()))  // The ball cannot provide enough points to get to k
                    )  // OR
                    || ball.is_leaf()
                        {
                            // The ball is a leaf, so we have to look at its points
                            if ball.is_singleton() {
                                // It's a singleton, so just add non-center items with the precomputed distance
                                hits.extend(ball.subtree_items().iter().map(|&p| (p, d)));
                            } else {
                                // Not a singleton, so compute distances to all non-center items
                                // and add them to hits
                                hits.extend(ball.subtree_items().iter().map(|&p| (p, metric(query, p))));
                            }
                        } else {
                            // Not a leaf, so add children to candidates
                            let [left, right] = ball.children().unwrap_or_else(|| unreachable!("Ball is a parent"));

                            // Compute distances to child centers
                            let left_d = metric(query, left.center());
                            let right_d = metric(query, right.center());

                            // Push child centers to hits
                            hits.push((left.center(), left_d));
                            hits.push((right.center(), right_d));

                            // Add children to candidates with their theoretical max distances
                            acc_candidates.push((left, d_max(left, left_d)));
                            acc_candidates.push((right, d_max(right, right_d)));
                        }
                        acc_candidates
                    });
        }

        hits.items().collect()
    }
}

impl<I: Send + Sync, T: DistanceValue + Send + Sync> ParSearch<I, T> for KnnBfs {}

/// Returns the theoretical maximum distance from the query to a point in the cluster.
fn d_max<I, T: DistanceValue>(ball: &Ball<I, T>, d: T) -> T {
    ball.radius() + d
}

/// Returns those candidates that are needed to guarantee the k-nearest
/// neighbors.
fn filter_candidates<I, T: DistanceValue>(mut candidates: Vec<(&Ball<I, T>, T)>, k: usize) -> Vec<(&Ball<I, T>, T)> {
    let threshold_index = quick_partition(&mut candidates, k);
    let threshold = candidates[threshold_index].1;

    candidates
        .into_iter()
        .filter_map(|(ball, d)| {
            let diam = ball.radius() + ball.radius();
            let d_min = if d <= diam { T::zero() } else { d - diam };
            if d_min <= threshold {
                Some((ball, d))
            } else {
                None
            }
        })
        .collect()
}

/// The Quick Partition algorithm, which is a variant of the Quick Select
/// algorithm. It finds the k-th smallest element in a list of elements, while
/// also reordering the list so that all elements to the left of the k-th
/// smallest element are less than or equal to it, and all elements to the right
/// of the k-th smallest element are greater than or equal to it.
fn quick_partition<I, T: DistanceValue>(items: &mut [(&Ball<I, T>, T)], k: usize) -> usize {
    qps(items, k, 0, items.len() - 1)
}

/// The recursive helper function for the Quick Partition algorithm.
fn qps<I, T: DistanceValue>(items: &mut [(&Ball<I, T>, T)], k: usize, l: usize, r: usize) -> usize {
    if l >= r {
        core::cmp::min(l, r)
    } else {
        // Choose the pivot point
        let pivot = l + (r - l) / 2;
        let p = find_pivot(items, l, r, pivot);

        // Calculate the cumulative guaranteed cardinalities for the first p
        // `Cluster`s
        let cumulative_guarantees = items
            .iter()
            .take(p)
            .scan(0, |acc, (ball, _)| {
                *acc += ball.cardinality();
                Some(*acc)
            })
            .collect::<Vec<_>>();

        // Calculate the guaranteed cardinality of the p-th `Cluster`
        let guaranteed_p = if p > 0 { cumulative_guarantees[p - 1] } else { 0 };

        match guaranteed_p.cmp(&k) {
            core::cmp::Ordering::Equal => p,                      // Found the k-th smallest element
            core::cmp::Ordering::Less => qps(items, k, p + 1, r), // Need to look to the right
            core::cmp::Ordering::Greater => {
                // The `Cluster` just before the p-th might be the one we need
                let guaranteed_p_minus_one = if p > 1 { cumulative_guarantees[p - 2] } else { 0 };
                if p == 0 || guaranteed_p_minus_one < k {
                    p // Found the k-th smallest element
                } else {
                    // Need to look to the left
                    qps(items, k, l, p - 1)
                }
            }
        }
    }
}

/// Moves pivot point and swaps elements around so that all elements to left
/// of pivot are less than or equal to pivot and all elements to right of pivot
/// are greater than pivot.
fn find_pivot<I, T: DistanceValue>(items: &mut [(&Ball<I, T>, T)], l: usize, r: usize, pivot: usize) -> usize {
    // Move pivot to the end
    items.swap(pivot, r);

    // Partition around pivot
    let (mut a, mut b) = (l, l);
    // Invariant: a <= b <= r
    while b < r {
        // If the current element is less than the pivot, swap it with the
        // element at a and increment a.
        if items[b].1 < items[r].1 {
            items.swap(a, b);
            a += 1;
        }
        // Increment b
        b += 1;
    }

    // Move pivot to its final position
    items.swap(a, r);

    a
}
