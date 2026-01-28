//! Compression and Decompression of `Cluster`s with appropriate generic constraints.

use rayon::prelude::*;

use crate::{
    Cluster,
    compress::{Codec, MaybeCodec},
    tree::AnnotatedItems,
};

impl<Id, I, T, A> Cluster<T, AnnotatedItems<Id, MaybeCodec<I>, A>>
where
    I: Codec,
{
    /// Gets a mutable reference to the center item, whether compressed or not.
    const fn center_mut(&mut self) -> &mut MaybeCodec<I> {
        &mut self.annotation_mut().center.1
    }

    /// Compresses all items in the cluster and its descendants.
    pub fn compress_all(mut self) -> Self {
        #[expect(unsafe_code)]
        // SAFETY: We replace the annotation at the end of the function.
        let AnnotatedItems {
            center,
            mut non_center,
            annotation,
        } = unsafe { core::ptr::read(&raw const self.annotation) };

        let (center_id, center) = center;
        let MaybeCodec::Original(center) = center else {
            unreachable!("Center item must be in original form when compressing");
        };

        self.children = self.children.map(|(mut children, span)| {
            children = children
                .into_iter()
                .map(|mut child| {
                    child = child.compress_all();
                    child.center_mut().encode_with(&center);
                    child
                })
                .collect();
            (children, span)
        });

        non_center = non_center.map(|non_centers| {
            non_centers
                .into_iter()
                .map(|(id, mut item)| {
                    item.encode_with(&center);
                    (id, item)
                })
                .collect()
        });

        self.set_annotation(AnnotatedItems {
            center: (center_id, MaybeCodec::Original(center)),
            non_center,
            annotation,
        });

        self
    }

    /// Decompresses all items in the cluster and its descendants.
    pub fn decompress_all(self) -> Self {
        todo!()
    }
}

impl<Id, I, T, A> Cluster<T, AnnotatedItems<Id, MaybeCodec<I>, A>>
where
    Id: Send,
    I: Codec + Send + Sync,
    I::Compressed: Send,
    T: Send,
    A: Send,
{
    /// Parallel version of [`Self::compress_all`].
    pub fn par_compress_all(mut self) -> Self {
        #[expect(unsafe_code)]
        // SAFETY: We replace the annotation at the end of the function.
        let AnnotatedItems {
            center,
            mut non_center,
            annotation,
        } = unsafe { core::ptr::read(&raw const self.annotation) };

        let (center_id, center) = center;
        let MaybeCodec::Original(center) = center else {
            unreachable!("Center item must be in original form when compressing");
        };

        self.children = self.children.map(|(mut children, span)| {
            children = children
                .into_par_iter()
                .map(|mut child| {
                    child = child.par_compress_all();
                    child.center_mut().encode_with(&center);
                    child
                })
                .collect();
            (children, span)
        });

        non_center = non_center.map(|non_centers| {
            non_centers
                .into_par_iter()
                .map(|(id, mut item)| {
                    item.encode_with(&center);
                    (id, item)
                })
                .collect()
        });

        self.set_annotation(AnnotatedItems {
            center: (center_id, MaybeCodec::Original(center)),
            non_center,
            annotation,
        });

        self
    }

    /// Parallel version of [`Self::decompress_all`].
    pub fn par_decompress_all(self) -> Self {
        todo!()
    }
}
