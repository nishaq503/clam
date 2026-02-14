//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
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

    fn search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        let root_radius = tree.root().radius();
        let d_root = tree.items[0].1.distance_to_query(query, &tree.metric)?;
        // Check to see if there is any overlap with the root
        if d_root > self.0 + root_radius {
            return Ok(Vec::new()); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, 0, root_radius)]; // (distance to cluster center, cluster index, cluster radius)
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
                tree.decompress_subtree(id)?;
                let indices = tree.get_cluster(id)?.subtree_indices();
                let distances = indices
                    .map(|i| tree.items[i].1.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                    .collect::<Result<Vec<_>, _>>()?;
                hits.extend(distances);
            } else if let Some(child_centers) = tree.decompress_child_centers(id)? {
                // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                let distances = child_centers
                    .iter()
                    .map(|&cid| {
                        tree.get_cluster(cid)
                            .map(|c| (cid, c.radius))
                            .and_then(|(cid, radius)| tree.items[cid].1.distance_to_query(query, &tree.metric).map(|d_child| (cid, d_child, radius)))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for &(cid, d_child, radius) in &distances {
                    if d_child <= self.0 + radius {
                        // This child cluster overlaps with the query ball, so we add it to the frontier
                        frontier.push((d_child, cid, radius));
                    }
                }
            } else {
                // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                tree.decompress_subtree(id)?;
                let indices = tree.get_cluster(id)?.subtree_indices();
                let distances = indices
                    .map(|i| tree.items[i].1.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                    .collect::<Result<Vec<_>, _>>()?;
                hits.extend(distances);
            }
        }

        Ok(hits)
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
    fn par_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        let root_radius = tree.root().radius();
        let d_root = tree.items[0].1.distance_to_query(query, &tree.metric)?;
        // Check to see if there is any overlap with the root
        if d_root > self.0 + root_radius {
            return Ok(Vec::new()); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, 0, root_radius)]; // (distance to cluster center, cluster index, cluster radius)
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
                tree.par_decompress_subtree(id)?;
                let indices = tree.get_cluster(id)?.subtree_indices();
                let distances = indices
                    .into_par_iter()
                    .map(|i| tree.items[i].1.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                    .collect::<Result<Vec<_>, _>>()?;
                hits.extend(distances);
            } else if let Some(child_centers) = tree.par_decompress_child_centers(id)? {
                // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                let distances = child_centers
                    .par_iter()
                    .map(|&cid| {
                        tree.get_cluster(cid)
                            .map(|c| (cid, c.radius))
                            .and_then(|(cid, radius)| tree.items[cid].1.distance_to_query(query, &tree.metric).map(|d_child| (cid, d_child, radius)))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for &(cid, d_child, radius) in &distances {
                    if d_child <= self.0 + radius {
                        // This child cluster overlaps with the query ball, so we add it to the frontier
                        frontier.push((d_child, cid, radius));
                    }
                }
            } else {
                // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                tree.par_decompress_subtree(id)?;
                let indices = tree.get_cluster(id)?.subtree_indices();
                let distances = indices
                    .into_par_iter()
                    .map(|i| tree.items[i].1.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                    .collect::<Result<Vec<_>, _>>()?;
                hits.extend(distances);
            }
        }

        Ok(hits)
    }
}
