//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use rayon::prelude::*;

use crate::{
    Cluster, DistanceValue, Tree,
    cakes::{ParSearch, Search},
};

/// Ranged Nearest Neighbors search using the CHESS algorithm.
///
/// The field is the radius of the query ball to search within.
pub struct RnnChess<T: DistanceValue>(pub T);

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for RnnChess<T>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn name(&self) -> String {
        format!("RnnChess(radius={})", self.0)
    }

    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let (mut hits, subsumed, straddlers) = tree_search(tree, &tree.root, query, self.0);

        // Add all items from fully subsumed clusters
        hits.extend(
            subsumed
                .into_iter()
                .flat_map(|cluster| tree.distances_to_items_in_subtree(query, cluster)),
        );

        // Check all items from straddling clusters
        hits.extend(straddlers.into_iter().flat_map(|cluster| {
            tree.distances_to_items_in_subtree(query, cluster)
                .into_iter()
                .filter(|(_, dist)| *dist <= self.0)
        }));

        hits
    }
}

impl<Id, I, T, A, M> ParSearch<Id, I, T, A, M> for RnnChess<T>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let (mut hits, subsumed, straddlers) = par_tree_search(tree, &tree.root, query, self.0);

        // Add all items from fully subsumed clusters
        hits.par_extend(
            subsumed
                .into_par_iter()
                .flat_map(|cluster| tree.par_distances_to_items_in_subtree(query, cluster)),
        );

        // Check all items from straddling clusters
        hits.par_extend(straddlers.into_par_iter().flat_map(|cluster| {
            tree.par_distances_to_items_in_subtree(query, cluster)
                .into_par_iter()
                .filter(|(_, dist)| *dist <= self.0)
        }));

        hits
    }
}

/// Perform coarse-grained tree search.
///
/// # Arguments
///
/// - `cluster` - The current cluster in the tree.
/// - `metric` - The distance metric function.
/// - `items` - The items in the tree.
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
pub fn tree_search<'a, Id, I, T, A, M>(
    tree: &'a Tree<Id, I, T, A, M>,
    cluster: &'a Cluster<T, A>,
    query: &I,
    radius: T,
) -> (Vec<(usize, T)>, Vec<&'a Cluster<T, A>>, Vec<&'a Cluster<T, A>>)
where
    T: DistanceValue + 'a,
    M: Fn(&I, &I) -> T,
{
    let center_dist = tree.distance_to_center(query, cluster);

    if center_dist > cluster.radius() + radius {
        // No overlapping volume between the query cluster and this cluster
        return (Vec::new(), Vec::new(), Vec::new());
    }

    if radius >= center_dist + cluster.radius() {
        // This cluster is fully contained within the query cluster
        return (vec![(cluster.center_index(), center_dist)], vec![cluster], Vec::new());
    }

    // This cluster overlaps the query cluster but is not fully contained.

    // Check whether our own center is within the query cluster
    let mut centers = if center_dist <= radius {
        vec![(cluster.center_index(), center_dist)]
    } else {
        Vec::new()
    };
    let mut subsumed = Vec::new();
    let mut straddlers = Vec::new();

    match cluster.children() {
        None => (centers, Vec::new(), vec![cluster]), // Leaf cluster
        Some(children) => {
            // Recurse into children
            for child in children {
                let (child_centers, child_subsumed, child_straddlers) = tree_search(tree, child, query, radius);
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
pub fn par_tree_search<'a, Id, I, T, A, M>(
    tree: &'a Tree<Id, I, T, A, M>,
    cluster: &'a Cluster<T, A>,
    query: &I,
    radius: T,
) -> (Vec<(usize, T)>, Vec<&'a Cluster<T, A>>, Vec<&'a Cluster<T, A>>)
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync + 'a,
    M: Fn(&I, &I) -> T + Send + Sync,
    A: Send + Sync,
{
    let center_dist = tree.distance_to_center(query, cluster);

    if center_dist > cluster.radius() + radius {
        // No overlapping volume between the query cluster and this cluster
        return (Vec::new(), Vec::new(), Vec::new());
    }

    if radius > center_dist + cluster.radius() {
        // This cluster is fully contained within the query cluster
        return (vec![(cluster.center_index(), center_dist)], vec![cluster], Vec::new());
    }

    // This cluster overlaps the query cluster but is not fully contained.

    // Check whether our own center is within the query cluster
    let mut centers = if center_dist <= radius {
        vec![(cluster.center_index(), center_dist)]
    } else {
        Vec::new()
    };
    let mut subsumed = Vec::new();
    let mut straddlers = Vec::new();

    match cluster.children() {
        None => (centers, Vec::new(), vec![cluster]), // Leaf cluster
        Some(children) => {
            // Recurse into children
            let returns = children
                .par_iter()
                .map(|child| par_tree_search(tree, child, query, radius))
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
