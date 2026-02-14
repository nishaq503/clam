//! Compression and compressive search algorithms. Use the `codec` feature to enable this module.

// mod search;
mod tree;

/// How an item can be encoded into and decoded from a compressed representation that consumes less memory than the original item.
pub trait Codec {
    /// The type of the compressed representation.
    type Compressed;

    /// Encodes the target item in terms of itself.
    fn compress(&self, target: &Self) -> Self::Compressed;

    /// Decodes the compressed representation in terms of itself.
    #[must_use]
    fn decompress(&self, compressed: &Self::Compressed) -> Self;

    /// Returns the number of bytes in the compressed representation.
    fn compressed_size(compressed: &Self::Compressed) -> usize;

    /// Returns the number of bytes in the original item.
    fn original_size(&self) -> usize;
}

/// An item that might be stored in a compressed form.
pub enum MaybeCompressed<I: Codec> {
    /// The original item.
    Original(I),
    /// The compressed representation of the item.
    Compressed(I::Compressed),
}

impl<I: Codec> MaybeCompressed<I> {
    /// Returns the original item if the item is stored in its original form, and None otherwise.
    pub(crate) fn take_original(self) -> Option<I> {
        match self {
            Self::Original(item) => Some(item),
            Self::Compressed(_) => None,
        }
    }

    /// Returns a reference to the original item if the item is stored in its original form, and None otherwise.
    pub const fn original(&self) -> Option<&I> {
        match self {
            Self::Original(item) => Some(item),
            Self::Compressed(_) => None,
        }
    }

    /// Returns a reference to the compressed representation of the item if the item is stored in its compressed form, and None otherwise.
    pub const fn compressed(&self) -> Option<&I::Compressed> {
        match self {
            Self::Original(_) => None,
            Self::Compressed(compressed) => Some(compressed),
        }
    }

    /// Returns the number of bytes required to store the item.
    pub fn size(&self) -> usize {
        match self {
            Self::Original(item) => item.original_size(),
            Self::Compressed(compressed) => I::compressed_size(compressed),
        }
    }

    /// Returns the distance from the query to this item if it is in its original form, and None otherwise.
    pub fn distance_to_query<T, M>(&self, query: &I, metric: &M) -> Option<T>
    where
        M: Fn(&I, &I) -> T,
    {
        self.original().map(|item| metric(item, query))
    }
}
