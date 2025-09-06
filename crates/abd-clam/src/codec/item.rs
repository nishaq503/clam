//! An item that has been compressed using an `Encoder` and can be decompressed during `CompressiveSearch`.

use super::{Decoder, Encoder};

/// An item or a reference to that item.
pub enum ItemOrRef<'a, I> {
    /// An owned item.
    Item(I),
    /// A reference to an item.
    Ref(&'a I),
}

impl<I> AsRef<I> for ItemOrRef<'_, I> {
    fn as_ref(&self) -> &I {
        match self {
            ItemOrRef::Item(item) => item,
            ItemOrRef::Ref(item) => item,
        }
    }
}

impl<I> ItemOrRef<'_, I> {
    /// Compute the distance between two `ItemOrRef`s using the provided metric.
    pub fn distance_to<T, M: Fn(&I, &I) -> T>(&self, other: &Self, metric: &M) -> T {
        metric(self.as_ref(), other.as_ref())
    }
}

/// An item that has been compressed using an `Encoder` and can be decompressed during `CompressiveSearch`.
pub enum CodecItem<I, Enc: Encoder<I, Dec> + ?Sized, Dec: Decoder<I, Enc> + ?Sized> {
    /// An uncompressed item.
    Uncompressed(I),
    /// A compressed delta against a reference item.
    Delta(Enc::Output),
}

impl<I, Enc: Encoder<I, Dec> + ?Sized, Dec: Decoder<I, Enc> + ?Sized> CodecItem<I, Enc, Dec> {
    /// Create a new `CodecItem` from an uncompressed item.
    pub const fn new_uncompressed(item: I) -> Self {
        Self::Uncompressed(item)
    }

    /// Create a new `CodecItem` from a delta.
    pub const fn new_delta(delta: Enc::Output) -> Self {
        Self::Delta(delta)
    }

    /// Decode the item without consuming it, using the provided decoder and return the decoded item.
    pub fn decode<'a>(&'a self, decoder: &Dec, reference: Option<&'a I>) -> ItemOrRef<'a, I> {
        match self {
            Self::Uncompressed(item) => ItemOrRef::Ref(item),
            Self::Delta(delta) => ItemOrRef::Item(reference.map_or_else(
                || decoder.decode_root(delta),
                |reference| decoder.decode(delta, reference),
            )),
        }
    }
}
