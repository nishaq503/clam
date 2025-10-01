//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use rayon::prelude::*;

use crate::{
    cakes::{BatchedSearch, Search},
    Cluster, DistanceValue,
};

/// Ranged Nearest Neighbors search using the CHESS algorithm.
///
/// The field is the radius of the query ball to search within.
pub struct RnnChess<T: DistanceValue>(pub T);

impl<T: DistanceValue> std::fmt::Display for RnnChess<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RnnChess(radius={})", self.0)
    }
}

impl<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A> Search<Id, I, T, M, A> for RnnChess<T> {
    fn search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)> {
        let (mut hits, subsumed, straddlers) = tree_search(root, metric, query, self.0);

        // Add all items from fully subsumed clusters
        hits.extend(subsumed.into_iter().flat_map(|cluster| {
            cluster
                .subtree_items()
                .into_iter()
                .map(|(id, item)| (id, item, metric(item, query)))
        }));

        // Check all items from straddling clusters
        hits.extend(straddlers.into_iter().flat_map(|cluster| {
            cluster.subtree_items().into_iter().filter_map(|(id, item)| {
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

    fn par_search<'a>(&self, root: &'a Cluster<Id, I, T, A>, metric: &M, query: &I) -> Vec<(&'a Id, &'a I, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        M: Send + Sync,
        A: Send + Sync,
    {
        let (mut hits, subsumed, straddlers) = par_tree_search(root, metric, query, self.0);

        // Add all items from fully subsumed clusters
        hits.extend(
            subsumed
                .into_par_iter()
                .flat_map(|cluster| {
                    cluster
                        .subtree_items()
                        .into_par_iter()
                        .map(|(id, item)| (id, item, metric(item, query)))
                })
                .collect::<Vec<_>>(),
        );

        // Check all items from straddling clusters
        hits.extend(
            straddlers
                .into_par_iter()
                .flat_map(|cluster| {
                    cluster.subtree_items().into_par_iter().filter_map(|(id, item)| {
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

        hits
    }
}

impl<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T, A> BatchedSearch<Id, I, T, M, A> for RnnChess<T> {}

/// Perform coarse-grained tree search.
///
/// # Arguments
///
/// - `cluster` - The root of the tree to search.
/// - `metric` - The metric to use for distance calculations.
/// - `query` - The query to search around.
/// - `radius` - The radius to search within.
///
/// # Returns
///
/// A tuple of three elements:
///   - centers, and their distances from the query, that are within the query cluster.
///   - clusters that are fully subsumed by the query cluster.
///   - clusters that have overlapping volume with the query cluster but are not fully subsumed.
#[allow(clippy::type_complexity)]
pub fn tree_search<'a, Id, I, T, M, A>(
    cluster: &'a Cluster<Id, I, T, A>,
    metric: &M,
    query: &I,
    radius: T,
) -> (
    Vec<(&'a Id, &'a I, T)>,
    Vec<&'a Cluster<Id, I, T, A>>,
    Vec<&'a Cluster<Id, I, T, A>>,
)
where
    T: DistanceValue + 'a,
    M: Fn(&I, &I) -> T,
{
    let center = cluster.center();
    let center_dist = metric(center, query);

    if center_dist > cluster.radius() + radius {
        // No overlapping volume between the query cluster and this cluster
        return (Vec::new(), Vec::new(), Vec::new());
    }

    if radius > center_dist + cluster.radius() {
        // This cluster is fully contained within the query cluster
        return (
            vec![(cluster.center_id(), center, center_dist)],
            vec![cluster],
            Vec::new(),
        );
    }

    // This cluster overlaps the query cluster but is not fully contained.

    // Check whether our own center is within the query cluster
    let mut centers = if center_dist <= radius {
        vec![(cluster.center_id(), center, center_dist)]
    } else {
        Vec::new()
    };
    let mut subsumed = Vec::new();
    let mut straddlers = Vec::new();

    match cluster.children() {
        None => (centers, Vec::new(), vec![cluster]), // Leaf node
        Some(children) => {
            // Recurse into children
            for child in children {
                let (child_centers, child_subsumed, child_straddlers) = tree_search(child, metric, query, radius);
                centers.extend(child_centers);
                subsumed.extend(child_subsumed);
                straddlers.extend(child_straddlers);
            }
            (centers, subsumed, straddlers)
        }
    }
}

/// Parallel version of [`tree_search`](tree_search).
#[allow(clippy::type_complexity)]
pub fn par_tree_search<'a, Id, I, T, M, A>(
    cluster: &'a Cluster<Id, I, T, A>,
    metric: &M,
    query: &I,
    radius: T,
) -> (
    Vec<(&'a Id, &'a I, T)>,
    Vec<&'a Cluster<Id, I, T, A>>,
    Vec<&'a Cluster<Id, I, T, A>>,
)
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync + 'a,
    M: Fn(&I, &I) -> T + Send + Sync,
    A: Send + Sync,
{
    let center = cluster.center();
    let center_dist = metric(center, query);

    if center_dist > cluster.radius() + radius {
        // No overlapping volume between the query cluster and this cluster
        return (Vec::new(), Vec::new(), Vec::new());
    }

    if radius > center_dist + cluster.radius() {
        // This cluster is fully contained within the query cluster
        return (
            vec![(cluster.center_id(), center, center_dist)],
            vec![cluster],
            Vec::new(),
        );
    }

    // This cluster overlaps the query cluster but is not fully contained.

    // Check whether our own center is within the query cluster
    let mut centers = if center_dist <= radius {
        vec![(cluster.center_id(), center, center_dist)]
    } else {
        Vec::new()
    };
    let mut subsumed = Vec::new();
    let mut straddlers = Vec::new();

    match cluster.children() {
        None => (centers, Vec::new(), vec![cluster]), // Leaf node
        Some(children) => {
            // Recurse into children
            let returns = children
                .par_iter()
                .map(|child| par_tree_search(child, metric, query, radius))
                .collect::<Vec<_>>();

            for (child_centers, child_subsumed, child_straddlers) in returns {
                centers.extend(child_centers);
                subsumed.extend(child_subsumed);
                straddlers.extend(child_straddlers);
            }

            (centers, subsumed, straddlers)
        }
    }
}
