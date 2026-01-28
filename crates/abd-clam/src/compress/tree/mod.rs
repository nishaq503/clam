//! Extensions of the `Tree` struct assuming the items implement the `Codec` trait.

use rayon::prelude::*;

use crate::Tree;

use super::{Codec, MaybeCodec};

mod cluster;

impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    I: Codec,
{
    /// Returns the tree with compressed items.
    pub fn compress_all(self) -> Tree<Id, MaybeCodec<I>, T, A, M> {
        let (items, root, metric) = self.into_parts();
        let items = items.into_iter().map(|(id, item)| (id, MaybeCodec::Original(item))).collect();
        let (root, items) = root.annotate_with_items(items).compress_all().collect_items_from_annotations();
        Tree::from_parts(items, root, metric)
    }
}

impl<Id, I, T, A, M> Tree<Id, MaybeCodec<I>, T, A, M>
where
    I: Codec,
{
    /// Returns the tree with decompressed items.
    pub fn decompress_all(self) -> Tree<Id, I, T, A, M> {
        let (items, root, metric) = self.into_parts();
        let (root, items) = root.annotate_with_items(items).decompress_all().collect_items_from_annotations();
        let items = items
            .into_iter()
            .map(|(id, item)| match item {
                MaybeCodec::Original(item) => (id, item),
                MaybeCodec::Compressed(_) => unreachable!("All items should be decompressed at this point"),
            })
            .collect();

        Tree::from_parts(items, root, metric)
    }
}

impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    Id: Send,
    I: Codec + Send + Sync,
    I::Compressed: Send,
    T: Send,
    A: Send,
{
    /// Parallel version of [`Self::compress_all`].
    pub fn par_compress_all(self) -> Tree<Id, MaybeCodec<I>, T, A, M> {
        let (items, root, metric) = self.into_parts();
        let items = items.into_par_iter().map(|(id, item)| (id, MaybeCodec::Original(item))).collect();
        let (root, items) = root.par_annotate_with_items(items).par_compress_all().par_collect_items_from_annotations();
        Tree::from_parts(items, root, metric)
    }
}

impl<Id, I, T, A, M> Tree<Id, MaybeCodec<I>, T, A, M>
where
    Id: Send,
    I: Codec + Send + Sync,
    I::Compressed: Send,
    T: Send,
    A: Send,
{
    /// Parallel version of [`Self::decompress_all`].
    pub fn par_decompress_all(self) -> Tree<Id, I, T, A, M> {
        let (items, root, metric) = self.into_parts();
        let (root, items) = root.par_annotate_with_items(items).par_decompress_all().par_collect_items_from_annotations();
        let items = items
            .into_par_iter()
            .map(|(id, item)| match item {
                MaybeCodec::Original(item) => (id, item),
                MaybeCodec::Compressed(_) => unreachable!("All items should be decompressed at this point"),
            })
            .collect();

        Tree::from_parts(items, root, metric)
    }
}
