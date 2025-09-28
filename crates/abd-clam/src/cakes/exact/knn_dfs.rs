//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::cmp::Reverse;

use crate::{
    cakes::{d_min, ParSearch, Search},
    utils::SizedHeap,
    Cluster, DistanceValue,
};

/// K-Nearest Neighbor (KNN) search using the Depth-First Sieve algorithm.
pub struct KnnDfs(pub usize);

impl std::fmt::Display for KnnDfs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KnnDfs(k={})", self.0)
    }
}

impl<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A> Search<Id, I, T, M, A> for KnnDfs {
    fn search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        profi::prof!("KnnDfs::search");

        if self.0 > root.cardinality() {
            // If k is greater than the number of points in the tree, return all
            // items with their distances.
            return root.distances_to_all_items(query, metric);
        }

        let mut candidates = SizedHeap::<&'a Cluster<Id, I, T, A>, Reverse<(T, T)>>::new(None);
        let mut hits = SizedHeap::<(&'a Id, &'a I), T>::new(Some(self.0));

        let d = metric(query, root.center());
        hits.push(((root.center_id(), root.center()), d));
        candidates.push((root, Reverse((d_min(root, d), d))));

        while !candidates.is_empty() {
            profi::prof!("KnnDfs::search::loop");

            // Find the next leaf to process.
            let (leaf, d, _) = pop_till_leaf(query, metric, &mut candidates, &mut hits);
            // Process the leaf and update hits.
            leaf_into_hits(query, metric, &mut hits, leaf, d);

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates
                .peek()
                .map_or_else(T::min_value, |(_, &Reverse((d_min, _)))| d_min);
            if hits.is_full() && max_h < min_c {
                // The closest candidate cannot improve our hits, so we can stop.
                break;
            }
        }

        hits.items().map(|((id, item), d)| (id, item, d)).collect()
    }
}

impl<
        I: Send + Sync,
        Id: Send + Sync,
        T: DistanceValue + Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
        A: Send + Sync,
    > ParSearch<Id, I, T, M, A> for KnnDfs
{
}

/// Pop candidates until the top candidate is a leaf. Then pop and return that
/// leaf along with its minimum distance from the query.
///
/// The user must ensure that `candidates` is non-empty before calling this
/// function.
#[allow(clippy::type_complexity)]
pub fn pop_till_leaf<'a, Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A>(
    query: &I,
    metric: &M,
    candidates: &mut SizedHeap<&'a Cluster<Id, I, T, A>, Reverse<(T, T)>>,
    hits: &mut SizedHeap<(&'a Id, &'a I), T>,
) -> (&'a Cluster<Id, I, T, A>, T, usize) {
    profi::prof!("KnnDfs::pop_till_leaf");

    let mut distance_computations = 0;
    while candidates.peek().map_or_else(
        || unreachable!("`candidates` is non-empty."),
        |(cluster, _)| !cluster.is_leaf(),
    ) {
        profi::prof!("pop-while-not-leaf");

        candidates.pop().and_then(|(parent, _)| parent.children()).map_or_else(
            || unreachable!("Top candidate is a parent."),
            |children| {
                distance_computations += children.len();

                for child in children {
                    let d = metric(query, child.center());
                    hits.push(((child.center_id(), child.center()), d));
                    candidates.push((child, Reverse((d_min(child, d), d))));
                }
            },
        );
    }

    let (leaf, d) = candidates.pop().map_or_else(
        || unreachable!("`candidates` is non-empty."),
        |(leaf, Reverse((_, d)))| (leaf, d),
    );
    (leaf, d, distance_computations)
}

/// Given a leaf cluster, compute the distance from the query to each item in
/// the leaf and push them onto `hits`.
pub fn leaf_into_hits<'a, Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A>(
    query: &I,
    metric: &M,
    hits: &mut SizedHeap<(&'a Id, &'a I), T>,
    leaf: &'a Cluster<Id, I, T, A>,
    d: T,
) -> usize {
    profi::prof!("KnnDfs::leaf_into_hits");

    if leaf.is_singleton() {
        // A singleton leaf has zero radius, so all items in the leaf are
        // exactly `d` from the query.
        hits.extend(leaf.subtree_items().into_iter().map(|(id, item)| ((id, item), d)));
        0
    } else {
        // A non-singleton leaf may have non-zero radius, so we need to compute
        // the distance from the query to each item in the leaf.
        hits.extend(
            leaf.subtree_items()
                .into_iter()
                .map(|(id, item)| ((id, item), metric(query, item))),
        );
        leaf.cardinality() - 1 // We already knew the distance to the center.
    }
}
