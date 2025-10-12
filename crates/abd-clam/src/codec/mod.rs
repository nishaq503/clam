//! Compression, decompression, and compressive search with CLAM.

use crate::{Cluster, DistanceValue};

mod item;

pub use item::{CodecItem, ItemOrRef};

/// An `Encoder` encodes items into a compressed representation.
///
/// It can encode items either by themselves (the center of a root `Cluster`),
/// or as a delta against a reference item.
///
/// An `Encoder` is paired with a `Decoder` that can decode the encoded
/// representations back into items.
pub trait Encoder<I, Dec: Decoder<I, Self> + ?Sized> {
    /// The type of representation used by this encoder.
    type Output;

    /// Encode an item by itself without using a reference item.
    fn encode_root(&self, item: &I) -> Self::Output;

    /// Encode an item as a delta against a reference item.
    fn encode(&self, item: &I, reference: &I) -> Self::Output;
}

/// A `Decoder` decodes items from their compressed representation.
///
/// It can decode items either from their raw encoded representation,
/// or from a delta against a reference item.
pub trait Decoder<I, Enc: Encoder<I, Self> + ?Sized> {
    /// Decode an item from its byte representation.
    fn decode_root(&self, bytes: &Enc::Output) -> I;

    /// Decode an item from a delta against a reference item.
    fn decode(&self, delta: &Enc::Output, reference: &I) -> I;
}

/// A `CompressiveSearch` performs nearest neighbor search on a `Cluster` tree, decompressing items only when needed.
pub trait CompressiveSearch<Id, I, T, A, M, Enc, Dec>: std::fmt::Display
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    Enc: Encoder<I, Dec> + ?Sized,
    Dec: Decoder<I, Enc> + ?Sized,
{
    /// For a given query, return its nearest neighbors as a vector of tuples `(id, distance)`.
    fn search<'a>(
        &self,
        root: &'a Cluster<Id, CodecItem<I, Enc, Dec>, T, A>,
        metric: &M,
        query: &I,
        decoder: &Dec,
    ) -> Vec<(&'a Id, T)>;

    /// Parallel version of [`Search::search`].
    ///
    /// The default implementation offers no parallelism. This method should be overridden for algorithms that will actually benefit from parallelism when
    /// searching for a single query.
    fn par_search<'a>(
        &self,
        root: &'a Cluster<Id, CodecItem<I, Enc, Dec>, T, A>,
        metric: &M,
        query: &I,
        decoder: &Dec,
    ) -> Vec<(&'a Id, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        M: Send + Sync,
        A: Send + Sync,
        Enc: Send + Sync,
        Dec: Send + Sync,
        Enc::Output: Send + Sync,
    {
        self.search(root, metric, query, decoder)
    }
}
