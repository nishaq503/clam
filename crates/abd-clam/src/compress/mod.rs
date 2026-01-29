//! Compression and compressive search algorithms. Use the `codec` feature to enable this module.

mod tree;

/// How an item can be encoded into and decoded from a compressed representation that consumes less memory than the original item.
pub trait Codec {
    /// The type of the compressed representation.
    type Compressed;

    /// Encodes the item in terms of a reference item.
    fn compress_with(&self, reference: &Self) -> Self::Compressed;

    /// Decodes the compressed representation in terms of a reference item.
    fn decompress_with(compressed: &Self::Compressed, reference: &Self) -> Self;
}

/// An item that might be stored in a compressed form.
pub enum MaybeCompressed<I: Codec> {
    /// The original item.
    Original(I),
    /// The compressed representation of the item.
    Compressed(I::Compressed),
}

impl<I: Codec> MaybeCompressed<I> {
    /// Encodes the item in terms of a reference item.
    ///
    /// If the item is in the `Compressed` variant, this method does nothing.
    pub fn compress_with(&mut self, reference: &I) {
        if let Self::Original(item) = self {
            let compressed = item.compress_with(reference);
            *self = Self::Compressed(compressed);
        }
    }

    /// Decodes the compressed representation in terms of a reference item.
    ///
    /// If the item is in the `Original` variant, this method does nothing.
    pub fn decompress_with(&mut self, reference: &I) {
        if let Self::Compressed(compressed) = self {
            let item = I::decompress_with(compressed, reference);
            *self = Self::Original(item);
        }
    }
}
