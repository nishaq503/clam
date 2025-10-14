//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use rayon::prelude::*;

use crate::{cakes::Search, DistanceValue, Node, Tree};

/// Ranged Nearest Neighbors search using the CHESS algorithm.
///
/// The field is the radius of the query ball to search within.
pub struct RnnChess<T: DistanceValue>(pub T);

impl<T: DistanceValue> std::fmt::Display for RnnChess<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RnnChess(radius={})", self.0)
    }
}

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for RnnChess<T>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();
        let metric = tree.metric();
        let items = tree.items();

        let (mut hits, subsumed, straddlers) = tree_search(root, metric, items, query, self.0);

        // Add all items from fully subsumed clusters
        hits.extend(
            subsumed
                .into_iter()
                .flat_map(|node| node.subtree_indices().map(|i| (i, metric(&items[i].1, query)))),
        );

        // Check all items from straddling clusters
        hits.extend(straddlers.into_iter().flat_map(|cluster| {
            cluster.subtree_indices().filter_map(|i| {
                let dist = metric(&items[i].1, query);
                if dist <= self.0 {
                    // Item is within the search radius so include it
                    Some((i, dist))
                } else {
                    None
                }
            })
        }));

        hits
    }

    fn par_search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        let root = tree.root();
        let metric = tree.metric();
        let items = tree.items();

        let (mut hits, subsumed, straddlers) = par_tree_search(root, metric, items, query, self.0);

        // Add all items from fully subsumed clusters
        hits.par_extend(subsumed.into_par_iter().flat_map(|node| {
            node.subtree_indices()
                .into_par_iter()
                .map(|i| (i, metric(&items[i].1, query)))
        }));

        // Check all items from straddling clusters
        hits.par_extend(straddlers.into_par_iter().flat_map(|cluster| {
            cluster.subtree_indices().into_par_iter().filter_map(|i| {
                let dist = metric(&items[i].1, query);
                if dist <= self.0 {
                    // Item is within the search radius so include it
                    Some((i, dist))
                } else {
                    None
                }
            })
        }));

        hits
    }
}

/// Perform coarse-grained tree search.
///
/// # Arguments
///
/// - `node` - The current node in the tree.
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
    node: &'a Node<T, A>,
    metric: &M,
    items: &[(Id, I)],
    query: &I,
    radius: T,
) -> (Vec<(usize, T)>, Vec<&'a Node<T, A>>, Vec<&'a Node<T, A>>)
where
    T: DistanceValue + 'a,
    M: Fn(&I, &I) -> T,
{
    let center = &items[node.center_index()].1;
    let center_dist = metric(center, query);

    if center_dist > node.radius() + radius {
        // No overlapping volume between the query cluster and this cluster
        return (Vec::new(), Vec::new(), Vec::new());
    }

    if radius > center_dist + node.radius() {
        // This cluster is fully contained within the query cluster
        return (vec![(node.center_index(), center_dist)], vec![node], Vec::new());
    }

    // This cluster overlaps the query cluster but is not fully contained.

    // Check whether our own center is within the query cluster
    let mut centers = if center_dist <= radius {
        vec![(node.center_index(), center_dist)]
    } else {
        Vec::new()
    };
    let mut subsumed = Vec::new();
    let mut straddlers = Vec::new();

    match node.children() {
        None => (centers, Vec::new(), vec![node]), // Leaf node
        Some(children) => {
            // Recurse into children
            for child in children {
                let (child_centers, child_subsumed, child_straddlers) = tree_search(child, metric, items, query, radius);
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
    node: &'a Node<T, A>,
    metric: &M,
    items: &[(Id, I)],
    query: &I,
    radius: T,
) -> (Vec<(usize, T)>, Vec<&'a Node<T, A>>, Vec<&'a Node<T, A>>)
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync + 'a,
    M: Fn(&I, &I) -> T + Send + Sync,
    A: Send + Sync,
{
    let center = &items[node.center_index()].1;
    let center_dist = metric(center, query);

    if center_dist > node.radius() + radius {
        // No overlapping volume between the query cluster and this cluster
        return (Vec::new(), Vec::new(), Vec::new());
    }

    if radius > center_dist + node.radius() {
        // This cluster is fully contained within the query cluster
        return (vec![(node.center_index(), center_dist)], vec![node], Vec::new());
    }

    // This cluster overlaps the query cluster but is not fully contained.

    // Check whether our own center is within the query cluster
    let mut centers = if center_dist <= radius {
        vec![(node.center_index(), center_dist)]
    } else {
        Vec::new()
    };
    let mut subsumed = Vec::new();
    let mut straddlers = Vec::new();

    match node.children() {
        None => (centers, Vec::new(), vec![node]), // Leaf node
        Some(children) => {
            // Recurse into children
            let returns = children
                .par_iter()
                .map(|child| par_tree_search(child, metric, items, query, radius))
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
