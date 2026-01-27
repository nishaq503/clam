//! Compression and Decompression of `Cluster`s with appropriate generic constraints.

use crate::{
    Cluster,
    compress::{Codec, MaybeCodec},
    tree::AnnotatedItems,
};

impl<Id, I, T, A> Cluster<T, AnnotatedItems<Id, MaybeCodec<I>, A>>
where
    I: Codec,
{
    /// Compresses all items in the cluster and its descendants.
    pub fn compress_all(self) -> Self {
        todo!()
    }

    /// Decompresses all items in the cluster and its descendants.
    pub fn decompress_all(self) -> Self {
        todo!()
    }
}

impl<Id, I, T, A> Cluster<T, AnnotatedItems<Id, MaybeCodec<I>, A>>
where
    Id: Send,
    I: Codec + Send,
    I::Compressed: Send,
    T: Send,
    A: Send,
{
    /// Parallel version of [`Self::compress_all`].
    pub fn par_compress_all(self) -> Self {
        todo!()
    }

    /// Parallel version of [`Self::decompress_all`].
    pub fn par_decompress_all(self) -> Self {
        todo!()
    }
}
