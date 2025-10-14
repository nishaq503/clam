//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::cmp::Reverse;

use crate::{
    cakes::{d_max, d_min, Search},
    utils::SizedHeap,
    Cluster, DistanceValue, Tree,
};

/// K-Nearest Neighbor (KNN) search using the Depth-First Sieve algorithm.
///
/// The field is the number of nearest neighbors to find (k).
pub struct KnnDfs(pub usize);

impl std::fmt::Display for KnnDfs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KnnDfs(k={})", self.0)
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnDfs {
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

        let mut candidates = SizedHeap::<&Cluster<T, A>, Reverse<(T, T, T)>>::new(None);
        let mut hits = SizedHeap::<usize, T>::new(Some(self.0));

        let d = metric(query, &items[root.center_index()].1);
        hits.push((root.center_index(), d));
        candidates.push((root, Reverse((d_min(root, d), d_max(root, d), d))));

        while !candidates.is_empty() {
            // Find the next leaf to process.
            let (leaf, d, _) = pop_till_leaf(query, metric, items, &mut candidates, &mut hits);
            // Process the leaf and update hits.
            leaf_into_hits(query, metric, items, &mut hits, leaf, d);

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            let min_c = candidates
                .peek()
                .map_or_else(T::min_value, |(_, &Reverse((d_min, _, _)))| d_min);
            if hits.is_full() && max_h < min_c {
                // The closest candidate cannot improve our hits, so we can stop.
                break;
            }
        }

        hits.take_items().collect()
    }
}

/// Pop candidates until the top candidate is a leaf. Then pop and return that
/// leaf along with its minimum distance from the query.
///
/// The user must ensure that `candidates` is non-empty before calling this
/// function.
#[allow(clippy::type_complexity)]
pub fn pop_till_leaf<'a, Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T>(
    query: &I,
    metric: &M,
    items: &[(Id, I)],
    candidates: &mut SizedHeap<&'a Cluster<T, A>, Reverse<(T, T, T)>>,
    hits: &mut SizedHeap<usize, T>,
) -> (&'a Cluster<T, A>, T, usize) {
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
                    let d = metric(query, &items[child.center_index()].1);
                    hits.push((child.center_index(), d));
                    candidates.push((child, Reverse((d_min(child, d), d_max(child, d), d))));
                }
            },
        );
    }

    let (leaf, d) = candidates.pop().map_or_else(
        || unreachable!("`candidates` is non-empty."),
        |(leaf, Reverse((_, _, d)))| (leaf, d),
    );
    (leaf, d, distance_computations)
}

/// Given a leaf cluster, compute the distance from the query to each item in
/// the leaf and push them onto `hits`.
///
/// Returns the number of distance computations performed, excluding the
/// distance to the center (which is already known).
pub fn leaf_into_hits<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T>(
    query: &I,
    metric: &M,
    items: &[(Id, I)],
    hits: &mut SizedHeap<usize, T>,
    leaf: &Cluster<T, A>,
    d: T,
) -> usize {
    profi::prof!("KnnDfs::leaf_into_hits");

    if leaf.is_singleton() {
        // A singleton leaf has zero radius, so all items in the leaf are
        // exactly `d` from the query.
        hits.extend(leaf.subtree_indices().map(|i| (i, d)));
        0
    } else {
        // A non-singleton leaf may have non-zero radius, so we need to compute
        // the distance from the query to each item in the leaf.
        hits.extend(leaf.subtree_indices().map(|i| (i, metric(query, &items[i].1))));
        leaf.cardinality() - 1 // We already knew the distance to the center.
    }
}
