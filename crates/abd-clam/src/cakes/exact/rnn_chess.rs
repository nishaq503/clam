//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
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
        let root = tree.root();
        let d_root = (tree.metric)(query, &tree.items[0].1); // root center index is always 0
        // Check to see if there is any overlap with the root
        if d_root > self.0 + root.radius() {
            return Vec::new(); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, root)];
        while !frontier.is_empty() {
            let (new_frontier, new_hits) = frontier
                .into_iter()
                .map(|(d, cluster)| {
                    if d <= self.0 + cluster.radius() {
                        // Overlapping volume

                        let mut local_hits = Vec::new();

                        if d <= self.0 {
                            // Center is within query radius
                            local_hits.push((cluster.center_index(), d));
                        }

                        if self.0 >= d + cluster.radius() {
                            // Fully subsumed cluster
                            for (i, (_, item)) in cluster.subtree_indices().zip(&tree.items[cluster.subtree_indices()]) {
                                local_hits.push((i, (tree.metric)(query, item)));
                            }
                            (Vec::new(), local_hits)
                        } else if let Some(children) = tree.children_of(cluster) {
                            let mut new_frontier = Vec::new();
                            for child in children {
                                let d_child = (tree.metric)(query, &tree.items[child.center_index()].1);
                                // Check children for overlap
                                if d_child <= self.0 + child.radius() {
                                    new_frontier.push((d_child, child));
                                }
                            }
                            (new_frontier, local_hits)
                        } else {
                            // Leaf cluster and not fully subsumed
                            for (i, d) in cluster
                                .subtree_indices()
                                .zip(&tree.items[cluster.subtree_indices()])
                                .filter_map(|(i, (_, item))| {
                                    let dist = (tree.metric)(query, item);
                                    if dist <= self.0 { Some((i, dist)) } else { None }
                                })
                            {
                                local_hits.push((i, d));
                            }
                            (Vec::new(), local_hits)
                        }
                    } else {
                        (Vec::new(), Vec::new())
                    }
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();

            frontier = new_frontier.into_iter().flatten().collect();
            hits.extend(new_hits.into_iter().flatten());
        }

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
        let root = tree.root();
        let d_root = (tree.metric)(query, &tree.items[0].1); // root center index is always 0
        // Check to see if there is any overlap with the root
        if d_root > self.0 + root.radius() {
            return Vec::new(); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, root)];
        while !frontier.is_empty() {
            let (new_frontier, new_hits) = frontier
                .into_par_iter()
                .map(|(d, cluster)| {
                    if d <= self.0 + cluster.radius() {
                        // Overlapping volume

                        let mut local_hits = Vec::new();

                        if d <= self.0 {
                            // Center is within query radius
                            local_hits.push((cluster.center_index(), d));
                        }

                        if self.0 >= d + cluster.radius() {
                            // Fully subsumed cluster
                            for (i, (_, item)) in cluster
                                .subtree_indices()
                                .into_par_iter()
                                .zip(&tree.items[cluster.subtree_indices()])
                                .collect::<Vec<_>>()
                            {
                                local_hits.push((i, (tree.metric)(query, item)));
                            }
                            (Vec::new(), local_hits)
                        } else if let Some(children) = tree.children_of(cluster) {
                            let mut new_frontier = Vec::new();
                            for child in children {
                                let d_child = (tree.metric)(query, &tree.items[child.center_index()].1);
                                // Check children for overlap
                                if d_child <= self.0 + child.radius() {
                                    new_frontier.push((d_child, child));
                                }
                            }
                            (new_frontier, local_hits)
                        } else {
                            // Leaf cluster and not fully subsumed
                            for (i, d) in cluster
                                .subtree_indices()
                                .into_par_iter()
                                .zip(&tree.items[cluster.subtree_indices()])
                                .filter_map(|(i, (_, item))| {
                                    let dist = (tree.metric)(query, item);
                                    if dist <= self.0 { Some((i, dist)) } else { None }
                                })
                                .collect::<Vec<_>>()
                            {
                                local_hits.push((i, d));
                            }
                            (Vec::new(), local_hits)
                        }
                    } else {
                        (Vec::new(), Vec::new())
                    }
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();

            frontier = new_frontier.into_iter().flatten().collect();
            hits.extend(new_hits.into_iter().flatten());
        }

        hits
    }
}
