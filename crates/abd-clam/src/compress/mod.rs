//! Compression and compressive search algorithms. Use the `codec` feature to enable this module.

mod tree;

/// How an item can be encoded into and decoded from a compressed representation that consumes less memory than the original item.
pub trait Codec {
    /// The type of the compressed representation.
    type Compressed;

    /// Encodes the item into its compressed representation.
    fn encode(&self) -> Self::Compressed;

    /// Encodes the item in terms of a reference item.
    fn encode_with(&self, reference: &Self) -> Self::Compressed;

    /// Decodes the compressed representation back into the original item.
    fn decode(compressed: &Self::Compressed) -> Self;

    /// Decodes the compressed representation in terms of a reference item.
    fn decode_with(compressed: &Self::Compressed, reference: &Self) -> Self;
}

/// An item that might be stored in a compressed form.
pub enum MaybeCodec<I: Codec> {
    /// The original item.
    Original(I),
    /// The compressed representation of the item.
    Compressed(I::Compressed),
}
