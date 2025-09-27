//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use rayon::prelude::*;

use crate::{Ball, DistanceValue};

use super::{ParSearch, Search};

/// Ranged Nearest Neighbors search using the CHESS algorithm.
pub struct RnnChess<T: DistanceValue>(pub T);

impl<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A> Search<Id, I, T, M, A> for RnnChess<T> {
    fn search<'a>(&self, root: &'a Ball<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        let (mut hits, subsumed, straddlers) = tree_search(root, metric, query, self.0);

        // Add all items from fully subsumed clusters
        hits.extend(subsumed.into_iter().flat_map(|ball| {
            ball.subtree_items()
                .into_iter()
                .map(|(id, item)| (id, item, metric(item, query)))
        }));

        // Check all items from straddling clusters
        hits.extend(straddlers.into_iter().flat_map(|ball| {
            ball.subtree_items().into_iter().filter_map(|(id, item)| {
                let dist = metric(item, query);
                if dist <= self.0 {
                    // Item is within the search radius so include it
                    Some((id, item, dist))
                } else {
                    None
                }
            })
        }));

        hits
    }
}

impl<
        I: Send + Sync,
        Id: Send + Sync,
        T: DistanceValue + Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
        A: Send + Sync,
    > ParSearch<Id, I, T, M, A> for RnnChess<T>
{
    fn par_search<'a>(&self, root: &'a Ball<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        profi::prof!("RnnChess::search");

        let (mut hits, subsumed, straddlers) = {
            profi::prof!("RnnChess::search::tree_search");

            par_tree_search(root, metric, query, self.0)
        };

        // Add all items from fully subsumed clusters
        {
            profi::prof!("RnnChess::search::subsumed_leaves");

            hits.extend(
                subsumed
                    .into_par_iter()
                    .flat_map(|ball| {
                        ball.subtree_items()
                            .into_par_iter()
                            .map(|(id, item)| (id, item, metric(item, query)))
                    })
                    .collect::<Vec<_>>(),
            );
        }

        // Check all items from straddling clusters
        {
            profi::prof!("RnnChess::search::straddler_leaves");

            hits.extend(
                straddlers
                    .into_par_iter()
                    .flat_map(|ball| {
                        ball.subtree_items().into_par_iter().filter_map(|(id, item)| {
                            let dist = metric(item, query);
                            if dist <= self.0 {
                                // Item is within the search radius so include it
                                Some((id, item, dist))
                            } else {
                                None
                            }
                        })
                    })
                    .collect::<Vec<_>>(),
            );
        }

        hits
    }
}

/// Perform coarse-grained tree search.
///
/// # Arguments
///
/// - `ball` - The root of the tree to search.
/// - `metric` - The metric to use for distance calculations.
/// - `query` - The query to search around.
/// - `radius` - The radius to search within.
///
/// # Returns
///
/// A tuple of three elements:
///   - centers, and their distances from the query, that are within the query ball.
///   - clusters that are fully subsumed by the query ball.
///   - clusters that have overlapping volume with the query ball but are not fully subsumed.
#[allow(clippy::type_complexity)]
pub fn tree_search<'a, Id, I, T, M, A>(
    ball: &'a Ball<Id, I, T, A>,
    metric: &M,
    query: &I,
    radius: T,
) -> (
    Vec<(&'a Id, &'a I, T)>,
    Vec<&'a Ball<Id, I, T, A>>,
    Vec<&'a Ball<Id, I, T, A>>,
)
where
    T: DistanceValue + 'a,
    M: Fn(&I, &I) -> T,
{
    let center = ball.center();
    let center_dist = metric(center, query);

    if center_dist > ball.radius() + radius {
        // No overlapping volume between the query ball and this cluster
        return (Vec::new(), Vec::new(), Vec::new());
    }

    if radius > center_dist + ball.radius() {
        // This cluster is fully contained within the query ball
        return (vec![(ball.center_id(), center, center_dist)], vec![ball], Vec::new());
    }

    // This cluster overlaps the query ball but is not fully contained.

    // Check whether our own center is within the query ball
    let mut centers = if center_dist <= radius {
        vec![(ball.center_id(), center, center_dist)]
    } else {
        Vec::new()
    };

    match ball.children() {
        None => (centers, Vec::new(), vec![ball]), // Leaf node
        Some([left, right]) => {
            // Recurse into children
            let (left_centers, mut subsumed, mut straddlers) = tree_search(left, metric, query, radius);
            let (right_centers, right_subsumed, right_straddlers) = tree_search(right, metric, query, radius);

            centers.extend(left_centers);
            centers.extend(right_centers);
            subsumed.extend(right_subsumed);
            straddlers.extend(right_straddlers);

            (centers, subsumed, straddlers)
        }
    }
}

/// Parallel version of [`tree_search`](tree_search).
#[allow(clippy::type_complexity)]
pub fn par_tree_search<'a, Id, I, T, M, A>(
    ball: &'a Ball<Id, I, T, A>,
    metric: &M,
    query: &I,
    radius: T,
) -> (
    Vec<(&'a Id, &'a I, T)>,
    Vec<&'a Ball<Id, I, T, A>>,
    Vec<&'a Ball<Id, I, T, A>>,
)
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync + 'a,
    M: Fn(&I, &I) -> T + Send + Sync,
    A: Send + Sync,
{
    let center = ball.center();
    let center_dist = metric(center, query);

    if center_dist > ball.radius() + radius {
        // No overlapping volume between the query ball and this cluster
        return (Vec::new(), Vec::new(), Vec::new());
    }

    if radius > center_dist + ball.radius() {
        // This cluster is fully contained within the query ball
        return (vec![(ball.center_id(), center, center_dist)], vec![ball], Vec::new());
    }

    // This cluster overlaps the query ball but is not fully contained.

    // Check whether our own center is within the query ball
    let mut centers = if center_dist <= radius {
        vec![(ball.center_id(), center, center_dist)]
    } else {
        Vec::new()
    };

    match ball.children() {
        None => (centers, Vec::new(), vec![ball]), // Leaf node
        Some([left, right]) => {
            // Recurse into children
            let ((left_centers, mut subsumed, mut straddlers), (right_centers, right_subsumed, right_straddlers)) =
                rayon::join(
                    || par_tree_search(left, metric, query, radius),
                    || par_tree_search(right, metric, query, radius),
                );

            centers.extend(left_centers);
            centers.extend(right_centers);
            subsumed.extend(right_subsumed);
            straddlers.extend(right_straddlers);

            (centers, subsumed, straddlers)
        }
    }
}
