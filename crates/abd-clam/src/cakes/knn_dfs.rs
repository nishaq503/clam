//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::cmp::Reverse;

use crate::{utils::SizedHeap, Ball, DistanceValue};

use super::{ParSearch, Search};

/// K-Nearest Neighbor (KNN) search using the Depth-First Sieve algorithm.
pub struct KnnDfs(pub usize);

impl<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T> Search<Id, I, T, M> for KnnDfs {
    fn search<'a>(&self, root: &'a Ball<Id, I, T>, metric: &M, query: &I) -> Vec<(&'a (Id, I), T)> {
        profi::prof!("KnnDfs::search");

        if self.0 > root.cardinality() {
            // If k is greater than the number of points in the tree, return all
            // items with their distances.
            return root.distances_to_all(query, metric);
        }

        let mut candidates = SizedHeap::<&'a Ball<Id, I, T>, Reverse<(T, T)>>::new(None);
        let mut hits = SizedHeap::<&(Id, I), T>::new(Some(self.0));

        let d = metric(query, &root.center().1);
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
            profi::prof!("KnnDfs::search::loop");

            // Find the next leaf to process.
            let (leaf, d) = pop_till_leaf(query, metric, &mut candidates, &mut hits);
            // Process the leaf and update hits.
            leaf_into_hits(query, metric, &mut hits, leaf, d);
        }

        hits.items().collect()
    }
}

impl<I: Send + Sync, Id: Send + Sync, T: DistanceValue + Send + Sync, M: Fn(&I, &I) -> T + Send + Sync>
    ParSearch<Id, I, T, M> for KnnDfs
{
}

/// The minimum possible distance from the query to any item in the ball.
fn d_min<Id, I, T: DistanceValue>(ball: &Ball<Id, I, T>, d: T) -> T {
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
#[allow(clippy::type_complexity)]
fn pop_till_leaf<'a, Id, I, T: DistanceValue, M: Fn(&I, &I) -> T>(
    query: &I,
    metric: &M,
    candidates: &mut SizedHeap<&'a Ball<Id, I, T>, Reverse<(T, T)>>,
    hits: &mut SizedHeap<&'a (Id, I), T>,
) -> (&'a Ball<Id, I, T>, T) {
    profi::prof!("KnnDfs::pop_till_leaf");

    while candidates.peek().map_or_else(
        || unreachable!("`candidates` is non-empty."),
        |(ball, _)| !ball.is_leaf(),
    ) {
        profi::prof!("pop-while-not-leaf");

        candidates.pop().and_then(|(parent, _)| parent.children()).map_or_else(
            || unreachable!("Top candidate is a parent."),
            |[left, right]| {
                let (d_left, d_right) = {
                    profi::prof!("child-distances");

                    let d_left = metric(query, &left.center().1);
                    let d_right = metric(query, &right.center().1);
                    (d_left, d_right)
                };
                {
                    profi::prof!("push-children");

                    // Push the child centers onto hits.
                    hits.push((left.center(), d_left));
                    hits.push((right.center(), d_right));

                    // Push the children onto candidates.
                    candidates.push((left, Reverse((d_min(left, d_left), d_left))));
                    candidates.push((right, Reverse((d_min(right, d_right), d_right))));
                }
            },
        );
    }

    candidates.pop().map_or_else(
        || unreachable!("`candidates` is non-empty."),
        |(leaf, Reverse((_, d)))| (leaf, d),
    )
}

/// Given a leaf ball, compute the distance from the query to each item in
/// the leaf and push them onto `hits`.
fn leaf_into_hits<'a, Id, I, T: DistanceValue, M: Fn(&I, &I) -> T>(
    query: &I,
    metric: &M,
    hits: &mut SizedHeap<&'a (Id, I), T>,
    leaf: &'a Ball<Id, I, T>,
    d: T,
) {
    profi::prof!("KnnDfs::leaf_into_hits");

    if leaf.is_singleton() {
        // A singleton leaf has zero radius, so all items in the leaf are
        // exactly `d` from the query.
        hits.extend(leaf.subtree_items().into_iter().map(|item| (item, d)));
    } else {
        // A non-singleton leaf may have non-zero radius, so we need to compute
        // the distance from the query to each item in the leaf.
        hits.extend(
            leaf.subtree_items()
                .into_iter()
                .map(|item| (item, metric(query, &item.1))),
        );
    }
}
