//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::cmp::Reverse;

use crate::{utils::SizedHeap, Ball, DistanceValue};

use super::{ParSearch, Search};

/// K-Nearest Neighbor (KNN) search using the Depth-First Sieve algorithm.
pub struct KnnDfs(pub usize);

impl<I, T: DistanceValue, M: Fn(&I, &I) -> T> Search<I, T, M> for KnnDfs {
    fn search<'a>(&self, root: &'a Ball<I, T>, metric: &M, query: &I) -> Vec<(&'a I, T)> {
        if self.0 > root.cardinality() {
            // If k is greater than the number of items in the tree, so we
            // just return all items in the tree.
            return root
                .all_items()
                .into_iter()
                .map(|item| (item, metric(query, item)))
                .collect();
        }

        let mut candidates = SizedHeap::<&'a Ball<I, T>, Reverse<(T, T)>>::new(None);
        let mut hits = SizedHeap::<&'a I, T>::new(Some(self.0));

        let d = metric(query, root.center());
        hits.push((root.center(), d));
        candidates.push((root, Reverse((d_min(root, d), d))));

        while !hits.is_full()  // We do not have enough hits.
            // or
            || (!candidates.is_empty()  // We have candidates.
                && hits  // and
                    .peek()  // the farthest hit so far
                    .map_or_else(|| unreachable!("`hits` is non-empty."), |(_, &d)| d)
                    >= candidates  // is farther than
                        .peek() // the theoretical closest candidate
                        .map_or_else(|| unreachable!("`candidates` is non-empty."), |(_, &Reverse((d_min, _)))| d_min))
        {
            // Find the next leaf to process.
            let (leaf, d) = pop_till_leaf(query, metric, &mut candidates, &mut hits);
            // Process the leaf and update hits.
            leaf_into_hits(query, metric, &mut hits, leaf, d);
        }

        hits.items().collect()
    }
}

impl<I: Send + Sync, T: DistanceValue + Send + Sync, M: Fn(&I, &I) -> T + Send + Sync> ParSearch<I, T, M> for KnnDfs {}

/// The minimum possible distance from the query to any item in the ball.
fn d_min<I, T: DistanceValue>(ball: &Ball<I, T>, d: T) -> T {
    if d < ball.radius() {
        T::zero()
    } else {
        d - ball.radius()
    }
}

/// Pop candidates until the top candidate is a leaf. Then pop and return that
/// leaf along with its minimum distance from the query.
///
/// The user must ensure that `candidates` is non-empty before calling this
/// function.
fn pop_till_leaf<'a, I, T: DistanceValue, M: Fn(&I, &I) -> T>(
    query: &I,
    metric: &M,
    candidates: &mut SizedHeap<&'a Ball<I, T>, Reverse<(T, T)>>,
    hits: &mut SizedHeap<&'a I, T>,
) -> (&'a Ball<I, T>, T) {
    while candidates.peek().map_or_else(
        || unreachable!("`candidates` is non-empty."),
        |(ball, _)| !ball.is_leaf(),
    ) {
        // Pop the parent candidate.
        let (parent, _) = candidates
            .pop()
            .unwrap_or_else(|| unreachable!("`candidates` is non-empty."));

        // Get the children of the parent.
        let [left, right] = parent
            .children()
            .unwrap_or_else(|| unreachable!("`parent` is not a leaf."));

        // Compute the distance from the query to each child center.
        let d_left = metric(query, left.center());
        let d_right = metric(query, right.center());

        // Push the child centers onto hits.
        hits.push((left.center(), d_left));
        hits.push((right.center(), d_right));

        // Push the children onto candidates.
        candidates.push((left, Reverse((d_min(left, d_left), d_left))));
        candidates.push((right, Reverse((d_min(right, d_right), d_right))));
    }

    candidates.pop().map_or_else(
        || unreachable!("`candidates` is non-empty."),
        |(leaf, Reverse((_, d)))| (leaf, d),
    )
}

/// Given a leaf ball, compute the distance from the query to each item in
/// the leaf and push them onto `hits`.
fn leaf_into_hits<'a, I, T: DistanceValue, M: Fn(&I, &I) -> T>(
    query: &I,
    metric: &M,
    hits: &mut SizedHeap<&'a I, T>,
    leaf: &'a Ball<I, T>,
    d: T,
) {
    if leaf.is_singleton() {
        // A singleton leaf has zero radius, so all items in the leaf are
        // exactly `d` from the query.
        hits.extend(leaf.subtree_items().into_iter().map(|item| (item, d)));
    } else {
        // A non-singleton leaf may have non-zero radius, so we need to compute
        // the distance from the query to each item in the leaf.
        hits.extend(leaf.subtree_items().into_iter().map(|item| (item, metric(query, item))));
    }
}
