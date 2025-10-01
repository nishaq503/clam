//! K-Nearest Neighbors (KNN) search using the Breadth-First Sieve algorithm.

#![expect(clippy::type_complexity)]

use crate::{
    cakes::{d_max, BatchedSearch, Search},
    utils::SizedHeap,
    Cluster, DistanceValue,
};

/// K-Nearest Neighbor (KNN) search using the Breadth-First Sieve algorithm.
///
/// The field is the number of nearest neighbors to find (k).
pub struct KnnBfs(pub usize);

impl std::fmt::Display for KnnBfs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KnnBfs(k={})", self.0)
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnBfs {
    fn search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        profi::prof!("KnnBfs::search");

        if self.0 > root.cardinality() {
            // If k is greater than the number of points in the tree, return all
            // items with their distances.
            return root.distances_to_all_items(query, metric);
        }

        let mut candidates = Vec::new();
        let mut hits = SizedHeap::<(&'a Id, &'a I), T>::new(Some(self.0));

        let d = metric(query, root.center());
        hits.push(((root.center_id(), root.center()), d));
        candidates.push((root, d_max(root, d)));

        while !candidates.is_empty() {
            let mut next_candidates = Vec::new();
            candidates = filter_candidates(candidates, self.0);

            for (cluster, d) in candidates {
                if (
                    next_candidates.len() <= self.0  // We still need more points to satisfy k, AND
                    && (cluster.cardinality() < (self.0 - next_candidates.len()))  // The cluster cannot provide enough points to get to k
                )  // OR
                || cluster.is_leaf()
                {
                    profi::prof!("KnnBfs::search::leaf");
                    // The cluster is a leaf, so we have to look at its points
                    if cluster.is_singleton() {
                        // It's a singleton, so just add non-center items with the precomputed distance
                        hits.extend(cluster.subtree_items().iter().map(|(id, item)| ((id, item), d)));
                    } else {
                        // Not a singleton, so compute distances to all non-center items
                        // and add them to hits
                        hits.extend(
                            cluster
                                .subtree_items()
                                .iter()
                                .map(|(id, item)| ((id, item), metric(query, item))),
                        );
                    }
                } else {
                    profi::prof!("KnnBfs::search::parent");

                    for child in cluster
                        .children()
                        .unwrap_or_else(|| unreachable!("Cluster is a parent"))
                    {
                        let d = metric(query, child.center());
                        hits.push(((child.center_id(), child.center()), d));
                        next_candidates.push((child.as_ref(), d_max(child, d)));
                    }
                }
            }

            candidates = next_candidates;
        }

        hits.take_items().map(|((id, item), d)| (id, item, d)).collect()
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> BatchedSearch<Id, I, T, A, M> for KnnBfs {}

/// Returns those candidates that are needed to guarantee the k-nearest
/// neighbors.
fn filter_candidates<Id, I, T: DistanceValue, A>(
    mut candidates: Vec<(&Cluster<Id, I, T, A>, T)>,
    k: usize,
) -> Vec<(&Cluster<Id, I, T, A>, T)> {
    profi::prof!("KnnBfs::filter_candidates");

    let threshold_index = quick_partition(&mut candidates, k);
    let threshold = candidates[threshold_index].1;

    candidates
        .into_iter()
        .filter_map(|(cluster, d)| {
            let diam = cluster.radius() + cluster.radius();
            let d_min = if d <= diam { T::zero() } else { d - diam };
            if d_min <= threshold {
                Some((cluster, d))
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
fn quick_partition<Id, I, T: DistanceValue, A>(items: &mut [(&Cluster<Id, I, T, A>, T)], k: usize) -> usize {
    profi::prof!("KnnBfs::quick_partition");

    qps(items, k, 0, items.len() - 1)
}

/// The recursive helper function for the Quick Partition algorithm.
fn qps<Id, I, T: DistanceValue, A>(items: &mut [(&Cluster<Id, I, T, A>, T)], k: usize, l: usize, r: usize) -> usize {
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
            .scan(0, |acc, (cluster, _)| {
                *acc += cluster.cardinality();
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
fn find_pivot<Id, I, T: DistanceValue, A>(
    items: &mut [(&Cluster<Id, I, T, A>, T)],
    l: usize,
    r: usize,
    pivot: usize,
) -> usize {
    profi::prof!("KnnBfs::find_pivot");

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
