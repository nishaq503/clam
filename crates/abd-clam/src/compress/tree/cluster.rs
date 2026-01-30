//! Compression and Decompression of `Cluster`s with appropriate generic constraints.

use rayon::prelude::*;

use crate::{
    Cluster,
    compress::{Codec, MaybeCompressed},
    tree::AnnotatedItems,
};

impl<Id, I, T, A> Cluster<T, AnnotatedItems<Id, MaybeCompressed<I>, A>>
where
    I: Codec,
{
    /// Gets a mutable reference to the center item, whether compressed or not.
    pub(crate) const fn center_mut(&mut self) -> &mut MaybeCompressed<I> {
        &mut self.annotation_mut().center.1
    }

    /// Compresses all items in the cluster and its descendants.
    pub fn compress_all(mut self) -> Self {
        // TODO(Najib): Write this with an iterative approach to avoid stack overflows on deep trees.

        #[expect(unsafe_code)]
        // SAFETY: We replace the annotation at the end of the function.
        let AnnotatedItems {
            center,
            mut non_center,
            annotation,
        } = unsafe { core::ptr::read(&raw const self.annotation) };

        let (center_id, center) = center;
        let MaybeCompressed::Original(center) = center else {
            unreachable!("Center item must be in original form when compressing");
        };

        self.children = self.children.map(|(mut children, child_center_indices, span)| {
            children = children
                .into_iter()
                .map(|mut child| {
                    child = child.compress_all();
                    child.center_mut().compress_with(&center);
                    child
                })
                .collect();
            (children, child_center_indices, span)
        });

        if let Some(non_centers) = &mut non_center {
            for (_, item) in non_centers {
                item.compress_with(&center);
            }
        }

        self.set_annotation(AnnotatedItems {
            center: (center_id, MaybeCompressed::Original(center)),
            non_center,
            annotation,
        });

        self
    }

    /// Decompresses all items in the cluster and its descendants.
    pub fn decompress_all(mut self) -> Self {
        // TODO(Najib): Write this with an iterative approach to avoid stack overflows on deep trees.

        #[expect(unsafe_code)]
        // SAFETY: We replace the annotation at the end of the function.
        let AnnotatedItems {
            center,
            mut non_center,
            annotation,
        } = unsafe { core::ptr::read(&raw const self.annotation) };

        let (center_id, center) = center;
        let MaybeCompressed::Original(center) = center else {
            unreachable!("Center item must be in original form when compressing");
        };

        self.children = self.children.map(|(mut children, child_center_indices, span)| {
            children = children
                .into_iter()
                .map(|mut child| {
                    child.center_mut().decompress_with(&center);
                    child = child.decompress_all();
                    child
                })
                .collect();
            (children, child_center_indices, span)
        });

        if let Some(non_centers) = &mut non_center {
            for (_, item) in non_centers {
                item.decompress_with(&center);
            }
        }

        self.set_annotation(AnnotatedItems {
            center: (center_id, MaybeCompressed::Original(center)),
            non_center,
            annotation,
        });

        self
    }
}

impl<Id, I, T, A> Cluster<T, AnnotatedItems<Id, MaybeCompressed<I>, A>>
where
    Id: Send,
    I: Codec + Send + Sync,
    I::Compressed: Send,
    T: Send,
    A: Send,
{
    /// Parallel version of [`Self::compress_all`].
    pub fn par_compress_all(mut self) -> Self {
        // TODO(Najib): Write this with an iterative approach to avoid stack overflows on deep trees.

        #[expect(unsafe_code)]
        // SAFETY: We replace the annotation at the end of the function.
        let AnnotatedItems {
            center,
            mut non_center,
            annotation,
        } = unsafe { core::ptr::read(&raw const self.annotation) };

        let (center_id, center) = center;
        let MaybeCompressed::Original(center) = center else {
            unreachable!("Center item must be in original form when compressing");
        };

        self.children = self.children.map(|(mut children, child_center_indices, span)| {
            children = children
                .into_par_iter()
                .map(|mut child| {
                    child = child.par_compress_all();
                    child.center_mut().compress_with(&center);
                    child
                })
                .collect();
            (children, child_center_indices, span)
        });

        if let Some(non_centers) = &mut non_center {
            non_centers.par_iter_mut().for_each(|(_, item)| item.compress_with(&center));
        }

        self.set_annotation(AnnotatedItems {
            center: (center_id, MaybeCompressed::Original(center)),
            non_center,
            annotation,
        });

        self
    }

    /// Parallel version of [`Self::decompress_all`].
    pub fn par_decompress_all(mut self) -> Self {
        // TODO(Najib): Write this with an iterative approach to avoid stack overflows on deep trees.

        #[expect(unsafe_code)]
        // SAFETY: We replace the annotation at the end of the function.
        let AnnotatedItems {
            center,
            mut non_center,
            annotation,
        } = unsafe { core::ptr::read(&raw const self.annotation) };

        let (center_id, center) = center;
        let MaybeCompressed::Original(center) = center else {
            unreachable!("Center item must be in original form when compressing");
        };

        self.children = self.children.map(|(mut children, child_center_indices, span)| {
            children = children
                .into_par_iter()
                .map(|mut child| {
                    child.center_mut().decompress_with(&center);
                    child = child.par_decompress_all();
                    child
                })
                .collect();
            (children, child_center_indices, span)
        });

        if let Some(non_centers) = &mut non_center {
            non_centers.par_iter_mut().for_each(|(_, item)| item.decompress_with(&center));
        }

        self.set_annotation(AnnotatedItems {
            center: (center_id, MaybeCompressed::Original(center)),
            non_center,
            annotation,
        });

        self
    }
}
