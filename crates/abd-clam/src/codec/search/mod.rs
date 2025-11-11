//! Compressive Search

use crate::{DistanceValue, Tree};

use super::{CodecItem, Decoder, Encoder};

/// A Compressive Search algorithm
pub trait CompressiveSearch<Id, I, T, A, M, Enc, Dec>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    Enc: Encoder<I, Dec> + ?Sized,
    Dec: Decoder<I, Enc> + ?Sized,
{
    /// The name of the compressive search algorithm.
    fn name(&self) -> String;

    /// For a given query, return its nearest neighbors as a vector of tuples `(id, distance)`.
    ///
    /// This should decode items as needed during search but leave them in their encoded state.
    fn search(&self, tree: &Tree<Id, CodecItem<I, Enc, Dec>, T, A, M>, query: &I) -> Vec<(usize, T)>;

    /// Same as [`CompressiveSearch::search`] but any items that are decoded during search will be left in their decoded state.
    fn search_mut(&mut self, tree: &Tree<Id, CodecItem<I, Enc, Dec>, T, A, M>, query: &I) -> Vec<(usize, T)>;
}
