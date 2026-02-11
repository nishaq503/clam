//! Ranged Nearest Neighbor (RNN) search with a naive linear scan.

use crate::{
    DistanceValue,
    codec::{CodecTree, CompressiveSearch, Decoder, Encoder},
};

/// Ranged Nearest Neighbor (RNN) search with a naive linear scan.
///
/// The field is the radius of the query ball to search within.
pub struct RnnLinear<T: DistanceValue>(pub T);

impl<Id, I, T, A, M, Enc, Dec> CompressiveSearch<Id, I, T, A, M, Enc, Dec> for RnnLinear<T>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    fn name(&self) -> String {
        format!("RnnLinear(radius={})", self.0)
    }

    fn search_mut(&self, tree: &mut CodecTree<Id, I, T, A, M, Enc, Dec>, query: &I) -> Vec<(usize, T)> {
        tree.distances_to_items_in_cluster_mut(query, tree.root.center_index)
            .into_iter()
            .filter_map(|(idx, dist)| if dist <= self.0 { Some((idx, dist)) } else { None })
            .collect()
    }
}
