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

impl<I: Codec> MaybeCodec<I> {
    /// Encodes the item into its compressed representation.
    ///
    /// If the item is in the `Compressed` variant, this method does nothing.
    pub fn encode(&mut self) {
        if let Self::Original(item) = self {
            let compressed = item.encode();
            *self = Self::Compressed(compressed);
        }
    }

    /// Encodes the item in terms of a reference item.
    ///
    /// If the item is in the `Compressed` variant, this method does nothing.
    pub fn encode_with(&mut self, reference: &I) {
        if let Self::Original(item) = self {
            let compressed = item.encode_with(reference);
            *self = Self::Compressed(compressed);
        }
    }

    /// Decodes the compressed representation back into the original item.
    ///
    /// If the item is in the `Original` variant, this method does nothing.
    pub fn decode(&mut self) {
        if let Self::Compressed(compressed) = self {
            let item = I::decode(compressed);
            *self = Self::Original(item);
        }
    }

    /// Decodes the compressed representation in terms of a reference item.
    ///
    /// If the item is in the `Original` variant, this method does nothing.
    pub fn decode_with(&mut self, reference: &I) {
        if let Self::Compressed(compressed) = self {
            let item = I::decode_with(compressed, reference);
            *self = Self::Original(item);
        }
    }
}
