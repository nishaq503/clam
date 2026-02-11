//! Compression, decompression, and compressive search with CLAM.

mod item;
mod search;
mod tree;

pub use item::CodecItem;
pub use search::{
    CompressiveSearch,
    exact::{RnnChess, RnnLinear},
};
// pub use search::{CompressiveSearch, exact::{KnnBfs, KnnBranch, KnnDfs, KnnLinear, KnnRrnn, RnnChess, RnnLinear}};
pub use tree::CodecTree;

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

    /// Encode an item by itself.
    fn encode_root(&self, item: &I) -> Self::Output;

    /// Encode an item as a delta against a reference item.
    fn encode(&self, item: &I, reference: &I) -> Self::Output;
}

/// A `Decoder` decodes items from their compressed representation.
///
/// It can decode items either from their raw encoded representation,
/// or from a delta against a reference item.
pub trait Decoder<I, Enc: Encoder<I, Self> + ?Sized> {
    /// Decode an item from its raw encoded representation.
    fn decode_root(&self, encoded: &Enc::Output) -> I;

    /// Decode an item from a delta against a reference item.
    fn decode(&self, delta: &Enc::Output, reference: &I) -> I;
}
