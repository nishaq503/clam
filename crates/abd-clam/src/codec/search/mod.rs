//! Compressive Search

use crate::{DistanceValue, codec::CodecTree};

use super::{Decoder, Encoder};

pub mod exact;

/// A Compressive Search algorithm
pub trait CompressiveSearch<Id, I, T, A, M, Enc, Dec>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    /// The name of the compressive search algorithm.
    fn name(&self) -> String;

    /// Same as [`CompressiveSearch::search`] but any items that are decoded during search will be left in their decoded state.
    fn search_mut(&self, tree: &mut CodecTree<Id, I, T, A, M, Enc, Dec>, query: &I) -> Vec<(usize, T)>;
}
