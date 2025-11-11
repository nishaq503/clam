//! An adaptation of the `Tree` for use in compression and compressive search.

use rayon::prelude::*;

use crate::{Cluster, DistanceValue, PartitionStrategy, Tree};

use super::{CodecItem, Decoder, Encoder};

/// An adaptation of the `Tree` for use in compression and compressive search.
#[must_use]
pub struct CodecTree<Id, I, T, A, M, Enc, Dec>
where
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    /// The items
    codec_items: Vec<(Id, CodecItem<I, Enc, Dec>)>,
    /// The underlying root cluster.
    root: Cluster<T, A>,
    /// The distance metric used.
    metric: M,
    /// The encoder used for compressing items.
    encoder: Enc,
    /// The decoder used for decompressing items.
    decoder: Dec,
}

impl<Id, I, T, A, M, Enc, Dec> core::fmt::Debug for CodecTree<Id, I, T, A, M, Enc, Dec>
where
    Id: core::fmt::Debug,
    I: core::fmt::Debug,
    T: core::fmt::Debug,
    A: core::fmt::Debug,
    M: core::fmt::Debug,
    Enc: Encoder<I, Dec> + core::fmt::Debug,
    Dec: Decoder<I, Enc> + core::fmt::Debug,
    Enc::Output: core::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodecTree")
            .field("codec_items", &self.codec_items)
            .field("root", &self.root)
            .field("metric", &self.metric)
            .field("encoder", &self.encoder)
            .field("decoder", &self.decoder)
            .finish()
    }
}

impl<Id, I, T, A, M, Enc, Dec> Clone for CodecTree<Id, I, T, A, M, Enc, Dec>
where
    Id: Clone,
    I: Clone,
    T: Clone,
    A: Clone,
    M: Clone,
    Enc: Encoder<I, Dec> + Clone,
    Dec: Decoder<I, Enc> + Clone,
    Enc::Output: Clone,
{
    fn clone(&self) -> Self {
        Self {
            codec_items: self.codec_items.clone(),
            root: self.root.clone(),
            metric: self.metric.clone(),
            encoder: self.encoder.clone(),
            decoder: self.decoder.clone(),
        }
    }
}

/// Various getters, setters, and some private helpers for `CodecTree`.
impl<Id, I, T, A, M, Enc, Dec> CodecTree<Id, I, T, A, M, Enc, Dec>
where
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    /// Get a reference to the codec items.
    pub fn codec_items(&self) -> &[(Id, CodecItem<I, Enc, Dec>)] {
        &self.codec_items
    }

    /// Get a reference to the underlying root cluster.
    pub const fn root(&self) -> &Cluster<T, A> {
        &self.root
    }

    /// Get a reference to the distance metric.
    pub const fn metric(&self) -> &M {
        &self.metric
    }

    /// Get a reference to the encoder.
    pub const fn encoder(&self) -> &Enc {
        &self.encoder
    }

    /// Get a reference to the decoder.
    pub const fn decoder(&self) -> &Dec {
        &self.decoder
    }

    /// Changes the metric used by the `CodecTree`.
    pub fn with_metric<N>(self, metric: N) -> CodecTree<Id, I, T, A, N, Enc, Dec>
    where
        N: Fn(&I, &I) -> T,
    {
        CodecTree {
            codec_items: self.codec_items,
            root: self.root,
            metric,
            encoder: self.encoder,
            decoder: self.decoder,
        }
    }

    /// Changes the encoder and decoder used by the `CodecTree`.
    pub fn with_codec<NewEnc, NewDec>(
        self,
        encoder: NewEnc,
        decoder: NewDec,
    ) -> CodecTree<Id, I, T, A, M, NewEnc, NewDec>
    where
        NewEnc: Encoder<I, NewDec>,
        NewDec: Decoder<I, NewEnc>,
        NewEnc::Output: From<Enc::Output>,
    {
        let codec_items = self
            .codec_items
            .into_iter()
            .map(|(id, item)| {
                let item = match item {
                    CodecItem::Uncompressed(item) => CodecItem::new_uncompressed(item),
                    CodecItem::Delta(delta) => {
                        let new_delta = NewEnc::Output::from(delta);
                        CodecItem::new_delta(new_delta)
                    }
                };
                (id, item)
            })
            .collect();
        CodecTree {
            codec_items,
            root: self.root,
            metric: self.metric,
            encoder,
            decoder,
        }
    }

    /// Decode all cluster centers along the branch from this cluster to the given cluster.
    fn decode_branch_to_cluster(&mut self, cluster: &Cluster<T, A>) {
        if matches!(&self.codec_items[cluster.center_index].1, CodecItem::Uncompressed(_)) {
            // The center is already uncompressed.
            return;
        }

        // Find the path from the root to the requested cluster.
        let path = cluster
            .path_to_cluster_containing(cluster.center_index)
            .unwrap_or_else(|_| unreachable!("The center index is never out of bounds."));

        // Decode the root item first.
        let mut path_iter = path.iter();
        let mut ref_index = path_iter.by_ref().next().map_or_else(
            || unreachable!("The path to the center should always have at least one element (the root)."),
            |root| root.center_index,
        );
        self.codec_items[ref_index].1.decode(&self.decoder, None);

        // Decode all centers along the path from the root to the center.
        for center_index in path_iter.map(|c| c.center_index) {
            match &self.codec_items[center_index].1 {
                CodecItem::Uncompressed(_) => {
                    // Already decoded;
                    continue;
                }
                CodecItem::Delta(delta) => {
                    let reference = match &self.codec_items[ref_index].1 {
                        CodecItem::Uncompressed(item) => item,
                        CodecItem::Delta(_) => unreachable!("Reference item should be uncompressed at this point."),
                    };
                    let item = self.decoder.decode(delta, reference);
                    self.codec_items[center_index].1 = CodecItem::new_uncompressed(item);
                }
            }
            ref_index = center_index;
        }
    }

    /// Decodes all non-center items in the given cluster using the center as reference, assuming the center is already decoded.
    fn decode_non_center_items(&mut self, cluster: &Cluster<T, A>) {
        let reference = match &self.codec_items[cluster.center_index].1 {
            CodecItem::Uncompressed(item) => item,
            CodecItem::Delta(_) => unreachable!("Center item should be uncompressed at this point."),
        };
        cluster.subtree_indices().for_each(|index| {
            // SAFETY: We have mutable access to the tree and we know that the center index is not in the subtree indices. Therefore, we are not taking
            // simultaneous mutable references to the same item.
            #[allow(unsafe_code)]
            unsafe {
                let (_, codec_item) = &mut *self.codec_items.as_ptr().cast_mut().add(index);
                codec_item.decode(&self.decoder, Some(reference));
            }
        });
    }

    /// Parallel version of [`Self::decode_non_center_items`].
    fn par_decode_non_center_items(&mut self, cluster: &Cluster<T, A>)
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
        Enc: Send + Sync,
        Enc::Output: Send + Sync,
        Dec: Send + Sync,
    {
        let reference = match &self.codec_items[cluster.center_index].1 {
            CodecItem::Uncompressed(item) => item,
            CodecItem::Delta(_) => unreachable!("Center item should be uncompressed at this point."),
        };
        cluster.subtree_indices().into_par_iter().for_each(|index| {
            // SAFETY: We have mutable access to the tree and we know that the center index is not in the subtree indices. Therefore, we are not taking
            // simultaneous mutable references to the same item.
            #[allow(unsafe_code)]
            unsafe {
                let (_, codec_item) = &mut *self.codec_items.as_ptr().cast_mut().add(index);
                codec_item.decode(&self.decoder, Some(reference));
            }
        });
    }
}

/// Constructors and methods for computing distances in `CodecTree`.
impl<Id, I, T, A, M, Enc, Dec> CodecTree<Id, I, T, A, M, Enc, Dec>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    /// Create a new `CodecTree`.
    ///
    /// # Errors
    ///
    /// See [`Self::new`] for possible errors.
    pub fn new<P, Ann>(
        items: Vec<(Id, I)>,
        metric: M,
        strategy: &PartitionStrategy<P>,
        annotator: &Ann,
        encoder: Enc,
        decoder: Dec,
    ) -> Result<Self, &'static str>
    where
        P: Fn(&Cluster<T, A>) -> bool,
        Ann: Fn(&Cluster<T, A>) -> Option<A>,
    {
        Tree::new(items, metric, strategy, annotator).map(|tree| Self::from_tree(tree, encoder, decoder))
    }

    /// Create a new `CodecTree` from a pre-existing `Tree`.
    pub fn from_tree(tree: Tree<Id, I, T, A, M>, encoder: Enc, decoder: Dec) -> Self {
        let Tree { items, root, metric } = tree;
        let mut codec_items = recursive_encode(&root, items, &encoder);
        codec_items[0].1.encode(&encoder, None); // Encode the root item.
        Self {
            codec_items,
            root,
            metric,
            encoder,
            decoder,
        }
    }

    /// Returns the distance between the query and the center of the given cluster, decoding items as necessary.
    pub fn distance_to_center_mut(&mut self, query: &I, cluster: &Cluster<T, A>) -> T {
        self.decode_branch_to_cluster(cluster);
        self.distance_to_center_decoded(query, cluster)
    }

    /// Returns the distances between the query item and all items in the given cluster, excluding the cluster's center, decoding items as necessary.
    pub fn distances_to_items_in_subtree_mut(&mut self, query: &I, cluster: &Cluster<T, A>) -> Vec<(usize, T)> {
        self.decode_branch_to_cluster(cluster);
        cluster
            .subtree_indices()
            .zip(self.distances_to_non_center_decoded(query, cluster))
            .collect()
    }

    /// Returns the distances between the query item and all items in the given cluster, including the cluster's center, decoding items as necessary.
    pub fn distances_to_items_in_cluster_mut(&mut self, query: &I, cluster: &Cluster<T, A>) -> Vec<(usize, T)> {
        self.decode_branch_to_cluster(cluster);
        self.decode_non_center_items(cluster);
        let distance_to_center = self.distance_to_center_decoded(query, cluster);
        let distances_to_non_center = self.distances_to_non_center_decoded(query, cluster);
        cluster
            .all_items_indices()
            .zip(core::iter::once(distance_to_center).chain(distances_to_non_center))
            .collect()
    }

    /// Returns the distance between the query and the center of the given cluster, assuming the center is already decoded.
    fn distance_to_center_decoded(&self, query: &I, cluster: &Cluster<T, A>) -> T {
        let center = match &self.codec_items[cluster.center_index].1 {
            CodecItem::Uncompressed(item) => item,
            CodecItem::Delta(_) => unreachable!("Center item should be uncompressed at this point."),
        };
        (self.metric)(query, center)
    }

    /// Computes the distances from the query to all non-center items in the given cluster, assuming the center and all items are already decoded.
    fn distances_to_non_center_decoded(&self, query: &I, cluster: &Cluster<T, A>) -> Vec<T> {
        let indices = cluster.subtree_indices();
        self.codec_items[indices]
            .iter()
            .map(|(_, codec_item)| {
                let item = match codec_item {
                    CodecItem::Uncompressed(item) => item,
                    CodecItem::Delta(_) => {
                        unreachable!("All items in the subtree should be uncompressed at this point.")
                    }
                };
                (self.metric)(query, item)
            })
            .collect()
    }
}

/// Parallelized constructors and methods for computing distances in `CodecTree`.
impl<Id, I, T, A, M, Enc, Dec> CodecTree<Id, I, T, A, M, Enc, Dec>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    Enc: Encoder<I, Dec> + Send + Sync,
    Enc::Output: Send + Sync,
    Dec: Decoder<I, Enc> + Send + Sync,
{
    /// Parallel version of [`Self::new`].
    ///
    /// # Errors
    ///
    /// See [`Tree::new`] for possible errors.
    pub fn par_new<P, Ann>(
        items: Vec<(Id, I)>,
        metric: M,
        strategy: &PartitionStrategy<P>,
        annotator: &Ann,
        encoder: Enc,
        decoder: Dec,
    ) -> Result<Self, &'static str>
    where
        P: Fn(&Cluster<T, A>) -> bool + Send + Sync,
        Ann: Fn(&Cluster<T, A>) -> Option<A> + Send + Sync,
    {
        Tree::new(items, metric, strategy, annotator).map(|tree| Self::par_from_tree(tree, encoder, decoder))
    }

    /// Parallel version of [`Self::from_tree`].
    pub fn par_from_tree(tree: Tree<Id, I, T, A, M>, encoder: Enc, decoder: Dec) -> Self {
        let Tree { items, root, metric } = tree;
        let mut codec_items = par_recursive_encode(&root, items, &encoder);
        codec_items[0].1.encode(&encoder, None); // Encode the root item.
        Self {
            codec_items,
            root,
            metric,
            encoder,
            decoder,
        }
    }

    /// Parallel version of [`Self::distances_to_items_in_cluster_mut`].
    ///
    /// The centers along the branch to the cluster are decoded in serial. Decoding of non-center items and the distance computations are then parallelized.
    pub fn par_distances_to_items_in_cluster_mut(&mut self, query: &I, cluster: &Cluster<T, A>) -> Vec<(usize, T)> {
        self.decode_branch_to_cluster(cluster);
        self.par_decode_non_center_items(cluster);
        let distance_to_center = self.distance_to_center_decoded(query, cluster);
        let distances_to_non_center = self.par_distances_to_non_center_decoded(query, cluster);
        cluster
            .all_items_indices()
            .zip(core::iter::once(distance_to_center).chain(distances_to_non_center))
            .collect()
    }

    /// Parallel version of [`Self::distances_to_non_center_decoded`].
    fn par_distances_to_non_center_decoded(&self, query: &I, cluster: &Cluster<T, A>) -> Vec<T> {
        let indices = cluster.subtree_indices();
        self.codec_items[indices]
            .par_iter()
            .map(|(_, codec_item)| {
                let item = match codec_item {
                    CodecItem::Uncompressed(item) => item,
                    CodecItem::Delta(_) => {
                        unreachable!("All items in the subtree should be uncompressed at this point.")
                    }
                };
                (self.metric)(query, item)
            })
            .collect()
    }
}

/// Recursively encode items in the tree using the provided encoder and decoder.
///
/// This assumes that the number of items matches the cardinality of the cluster.
fn recursive_encode<Id, I, T, A, Enc, Dec>(
    cluster: &Cluster<T, A>,
    mut items: Vec<(Id, I)>,
    encoder: &Enc,
) -> Vec<(Id, CodecItem<I, Enc, Dec>)>
where
    Enc: Encoder<I, Dec>,
    Dec: Decoder<I, Enc>,
{
    if let Some((children, _)) = &cluster.children {
        // Parent cluster: encode each child cluster recursively. Then encode each child center as a delta from the parent center.

        // Split items for each child cluster.
        let mut splits = children
            .iter()
            .rev()
            .map(|child| items.split_off(child.cardinality))
            .collect::<Vec<_>>();
        splits.reverse();

        // There should be exactly one item left, the center.
        let (center_id, center) = items
            .pop()
            .unwrap_or_else(|| unreachable!("This should be the center item."));

        // Encode each child cluster recursively.
        let codec_splits = children
            .iter()
            .zip(splits)
            .map(|(child, child_items)| {
                let mut codec_items = recursive_encode(child, child_items, encoder);
                codec_items[0].1.encode(encoder, Some(&center));
                codec_items
            })
            .collect::<Vec<_>>();

        // Do not compress the center item.
        core::iter::once((center_id, CodecItem::new_uncompressed(center)))
            .chain(codec_splits.into_iter().flatten())
            .collect()
    } else {
        // Leaf cluster: encode all non-center items as deltas from the center.

        // Take out the non-center items.
        let non_center_items = items.split_off(cluster.cardinality - 1);

        // There should be exactly one item left, the center.
        let (center_id, center) = items
            .pop()
            .unwrap_or_else(|| unreachable!("This should be the center item."));

        // Encode each non-center item as a delta from the center.
        let codec_items = non_center_items
            .into_iter()
            .map(|(id, item)| {
                let delta = encoder.encode(&item, &center);
                (id, CodecItem::new_delta(delta))
            })
            .collect::<Vec<_>>();

        // Do not compress the center item.
        core::iter::once((center_id, CodecItem::new_uncompressed(center)))
            .chain(codec_items)
            .collect()
    }
}

/// Parallel version of [`recursive_encode`].
fn par_recursive_encode<Id, I, T, A, Enc, Dec>(
    cluster: &Cluster<T, A>,
    mut items: Vec<(Id, I)>,
    encoder: &Enc,
) -> Vec<(Id, CodecItem<I, Enc, Dec>)>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: Send + Sync,
    A: Send + Sync,
    Enc: Encoder<I, Dec> + Send + Sync,
    Enc::Output: Send + Sync,
    Dec: Decoder<I, Enc> + Send + Sync,
{
    if let Some((children, _)) = &cluster.children {
        // Parent cluster: encode each child cluster recursively. Then encode each child center as a delta from the parent center.

        // Split items for each child cluster.
        let mut splits = children
            .iter()
            .rev()
            .map(|child| items.split_off(child.cardinality))
            .collect::<Vec<_>>();
        splits.reverse();

        // There should be exactly one item left, the center.
        let (center_id, center) = items
            .pop()
            .unwrap_or_else(|| unreachable!("This should be the center item."));

        // Encode each child cluster recursively.
        let codec_splits = children
            .par_iter()
            .zip(splits)
            .map(|(child, child_items)| {
                let mut codec_items = par_recursive_encode(child, child_items, encoder);
                codec_items[0].1.encode(encoder, Some(&center));
                codec_items
            })
            .collect::<Vec<_>>();

        // Do not compress the center item.
        core::iter::once((center_id, CodecItem::new_uncompressed(center)))
            .chain(codec_splits.into_iter().flatten())
            .collect()
    } else {
        // Leaf cluster: encode all non-center items as deltas from the center.

        // Take out the non-center items.
        let non_center_items = items.split_off(cluster.cardinality - 1);

        // There should be exactly one item left, the center.
        let (center_id, center) = items
            .pop()
            .unwrap_or_else(|| unreachable!("This should be the center item."));

        // Encode each non-center item as a delta from the center.
        let codec_items = non_center_items
            .into_par_iter()
            .map(|(id, item)| {
                let delta = encoder.encode(&item, &center);
                (id, CodecItem::new_delta(delta))
            })
            .collect::<Vec<_>>();

        // Do not compress the center item.
        core::iter::once((center_id, CodecItem::new_uncompressed(center)))
            .chain(codec_items)
            .collect()
    }
}
