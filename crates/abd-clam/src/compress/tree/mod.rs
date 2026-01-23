//! Extensions of the `Tree` struct assuming the items implement the `Codec` trait.

use crate::Tree;

use super::{Codec, MaybeCodec};

impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    I: Codec,
{
    /// Returns the tree with compressed items.
    pub fn compress_all(&self) -> Tree<Id, MaybeCodec<I>, T, A, M> {
        todo!()
    }
}

impl<Id, I, T, A, M> Tree<Id, MaybeCodec<I>, T, A, M>
where
    I: Codec,
{
    /// Returns the tree with decompressed items.
    pub fn decompress_all(&self) -> Tree<Id, I, T, A, M> {
        todo!()
    }
}
