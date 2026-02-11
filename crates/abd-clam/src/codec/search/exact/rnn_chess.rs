//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use core::ops::Range;

use crate::{
    DistanceValue,
    codec::{CodecTree, CompressiveSearch, Decoder, Encoder},
};

/// Ranged Nearest Neighbors search using the CHESS algorithm.
///
/// The field is the radius of the query ball to search within.
pub struct RnnChess<T: DistanceValue>(pub T);

impl<Id, I, T, A, M, Enc, Dec> CompressiveSearch<Id, I, T, A, M, Enc, Dec> for RnnChess<T>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    fn name(&self) -> String {
        format!("RnnChess(radius={})", self.0)
    }

    fn search_mut(&self, tree: &mut CodecTree<Id, I, T, A, M, Enc, Dec>, query: &I) -> Vec<(usize, T)> {
        let (mut hits, subsumed, straddlers) = compressive_tree_search(tree, tree.root.center_index(), query, self.0);

        // FIXME(Najib): This is re-computing distances to cluster centers. Optimize this.\

        // Add all items from fully subsumed clusters
        for indices in subsumed {
            hits.extend(tree.distances_to_items_in_cluster_mut(query, indices.start));
        }

        // Check all items from straddling clusters
        for cluster in straddlers {
            hits.extend(
                tree.distances_to_items_in_cluster_mut(query, cluster.start)
                    .into_iter()
                    .filter(|(_, dist)| *dist <= self.0),
            );
        }

        hits
    }
}

/// Perform coarse-grained tree search.
///
/// # Arguments
///
/// - `tree` - The codec tree to search.
/// - `center_index` - The index of the center item in the tree.
/// - `query` - The query to search around.
/// - `radius` - The radius to search within.
///
/// # Returns
///
/// A tuple of three elements:
///   - centers, and their distances from the query, that are within the query cluster.
///   - indices of clusters that are fully subsumed by the query cluster.
///   - indices of clusters that have overlapping volume with the query cluster but are not fully subsumed.
#[expect(clippy::type_complexity, unused_variables, clippy::needless_pass_by_ref_mut)]
pub fn compressive_tree_search<'a, Id, I, T, A, M, Enc, Dec>(
    tree: &'a mut CodecTree<Id, I, T, A, M, Enc, Dec>,
    center_index: usize,
    query: &I,
    radius: T,
) -> (Vec<(usize, T)>, Vec<Range<usize>>, Vec<Range<usize>>)
where
    T: DistanceValue + 'a,
    M: Fn(&I, &I) -> T,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    todo!()
}
