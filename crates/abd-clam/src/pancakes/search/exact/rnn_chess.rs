//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use rayon::prelude::*;

use crate::{
    Cluster, DistanceValue, Tree,
    pancakes::{Codec, MaybeCompressed},
};

use super::super::{CompressiveSearch, ParCompressiveSearch};

/// Ranged Nearest Neighbors search using the CHESS algorithm.
///
/// The field is the radius of the query ball to search within.
pub struct RnnChess<T: DistanceValue>(pub T);

impl<Id, I, T, A, M> CompressiveSearch<Id, I, T, A, M> for RnnChess<T>
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn name(&self) -> String {
        format!("RnnChess(radius={})", self.0)
    }

    fn search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();
        let d_root = tree.items[0]
            .1
            .distance_to_query(query, &tree.metric)
            .unwrap_or_else(|| unreachable!("The root center is never compressed."));
        // Check to see if there is any overlap with the root
        if d_root > self.0 + root.radius() {
            return Vec::new(); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, 0, root.radius())]; // (distance to cluster center, cluster index, cluster radius)
        while let Some((d, id, radius)) = frontier.pop() {
            if self.0 + radius < d {
                continue; // No overlap
            }
            // We have some overlap, so we need to check this cluster

            if d <= self.0 {
                // Center is within query radius
                hits.push((id, d));
            }

            if d + radius <= self.0 {
                // Fully subsumed cluster, so we will decompress the subtree and add all items in this subtree to the hits
                tree.decompress_subtree(id)
                    .unwrap_or_else(|err| unreachable!("Decompression should never fail during search: {err}"));
                if let Some(indices) = tree.cluster_map.get(&id).map(Cluster::subtree_indices) {
                    let distances = indices
                        .into_iter()
                        .map(|i| {
                            tree.items[i]
                                .1
                                .distance_to_query(query, &tree.metric)
                                .map(|d| (i, d))
                                .ok_or_else(|| "Decompressed items should never fail to compute distance.".to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap_or_else(|_| unreachable!("Decompressed items should never fail to compute distance."));
                    hits.extend(distances);
                }
            } else if let Some(child_centers) = tree
                .decompress_child_centers(id)
                .unwrap_or_else(|err| unreachable!("Decompression should never fail during search: {err}"))
            {
                // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                let distances = child_centers
                    .iter()
                    .map(|&cid| {
                        let d_child = tree.items[cid]
                            .1
                            .distance_to_query(query, &tree.metric)
                            .unwrap_or_else(|| unreachable!("Decompressed items should never fail to compute distance."));
                        let radius = tree
                            .cluster_map
                            .get(&cid)
                            .unwrap_or_else(|| unreachable!("Child cluster should always be in the cluster map."))
                            .radius;
                        (cid, d_child, radius)
                    })
                    .collect::<Vec<_>>();
                for &(cid, d_child, radius) in &distances {
                    if d_child <= self.0 + radius {
                        // This child cluster overlaps with the query ball, so we add it to the frontier
                        frontier.push((d_child, cid, radius));
                    }
                }
            } else {
                // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                tree.decompress_subtree(id)
                    .unwrap_or_else(|err| unreachable!("Decompression should never fail during search: {err}"));
                let indices = tree
                    .cluster_map
                    .get(&id)
                    .map_or_else(|| unreachable!("Cluster should always be in the cluster map."), Cluster::subtree_indices)
                    .collect::<Vec<_>>();
                let distances = indices
                    .iter()
                    .map(|&i| {
                        let item = &tree.items[i].1;
                        let dist = item
                            .distance_to_query(query, &tree.metric)
                            .unwrap_or_else(|| unreachable!("Decompressed items should never fail to compute distance."));
                        (i, dist)
                    })
                    .collect::<Vec<_>>();
                hits.extend(distances);
            }
        }

        hits
    }
}

impl<Id, I, T, A, M> ParCompressiveSearch<Id, I, T, A, M> for RnnChess<T>
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();
        let d_root = tree.items[0]
            .1
            .distance_to_query(query, &tree.metric)
            .unwrap_or_else(|| unreachable!("The root center is never compressed."));
        // Check to see if there is any overlap with the root
        if d_root > self.0 + root.radius() {
            return Vec::new(); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, 0, root.radius())]; // (distance to cluster center, cluster index, cluster radius)
        while let Some((d, id, radius)) = frontier.pop() {
            if self.0 + radius < d {
                continue; // No overlap
            }
            // We have some overlap, so we need to check this cluster

            if d <= self.0 {
                // Center is within query radius
                hits.push((id, d));
            }

            if d + radius <= self.0 {
                // Fully subsumed cluster, so we will decompress the subtree and add all items in this subtree to the hits
                tree.par_decompress_subtree(id)
                    .unwrap_or_else(|err| unreachable!("Decompression should never fail during search: {err}"));
                if let Some(indices) = tree.cluster_map.get(&id).map(Cluster::subtree_indices) {
                    let distances = indices
                        .into_par_iter()
                        .map(|i| {
                            tree.items[i]
                                .1
                                .distance_to_query(query, &tree.metric)
                                .map(|d| (i, d))
                                .ok_or_else(|| "Decompressed items should never fail to compute distance.".to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap_or_else(|_| unreachable!("Decompressed items should never fail to compute distance."));
                    hits.extend(distances);
                }
            } else if let Some(child_centers) = tree
                .par_decompress_child_centers(id)
                .unwrap_or_else(|err| unreachable!("Decompression should never fail during search: {err}"))
            {
                // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                let distances = child_centers
                    .par_iter()
                    .map(|&cid| {
                        let d_child = tree.items[cid]
                            .1
                            .distance_to_query(query, &tree.metric)
                            .unwrap_or_else(|| unreachable!("Decompressed items should never fail to compute distance."));
                        let radius = tree
                            .cluster_map
                            .get(&cid)
                            .unwrap_or_else(|| unreachable!("Child cluster should always be in the cluster map."))
                            .radius;
                        (cid, d_child, radius)
                    })
                    .collect::<Vec<_>>();
                for &(cid, d_child, radius) in &distances {
                    if d_child <= self.0 + radius {
                        // This child cluster overlaps with the query ball, so we add it to the frontier
                        frontier.push((d_child, cid, radius));
                    }
                }
            } else {
                // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                tree.par_decompress_subtree(id)
                    .unwrap_or_else(|err| unreachable!("Decompression should never fail during search: {err}"));
                let indices = tree
                    .cluster_map
                    .get(&id)
                    .map_or_else(|| unreachable!("Cluster should always be in the cluster map."), Cluster::subtree_indices)
                    .collect::<Vec<_>>();
                let distances = indices
                    .par_iter()
                    .map(|&i| {
                        let item = &tree.items[i].1;
                        let dist = item
                            .distance_to_query(query, &tree.metric)
                            .unwrap_or_else(|| unreachable!("Decompressed items should never fail to compute distance."));
                        (i, dist)
                    })
                    .collect::<Vec<_>>();
                hits.extend(distances);
            }
        }

        hits
    }
}
