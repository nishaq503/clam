//! A `Tree` of `Clusters` for use in CLAM.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::DistanceValue;

mod cluster;

pub use cluster::{BranchingFactor, Cluster, PartitionStrategy, SpanReductionFactor, lfd_estimate};

/// A tree structure used in CLAM for organizing items based on a given metric.
#[must_use]
#[derive(Clone, Debug)]
pub struct Tree<Id, I, T, A, M> {
    /// The items stored in the tree, each paired with its identifier.
    pub(crate) items: Vec<(Id, I)>,
    /// The root cluster of the tree.
    pub(crate) root: Cluster<T, A>,
    /// The metric used to compute distances between items.
    pub(crate) metric: M,
}

impl<I, T, M> Tree<usize, I, T, (), M>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    /// Creates a new `Tree` from the given items and metric.
    ///
    /// # Explanation
    ///
    /// This is a minimal constructor that assigns sequential integer IDs (starting from 0) to the items. It will also use the default
    /// [`PartitionStrategy`](PartitionStrategy).
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn new_minimal(items: Vec<I>, metric: M) -> Result<Self, &'static str> {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let mut items = items.into_iter().enumerate().collect::<Vec<_>>();
        let root = Cluster::new_root(&mut items, &metric, &PartitionStrategy::default(), &|_, _, _| None);

        Ok(Self { items, root, metric })
    }

    /// Parallel version of [`new_minimal`](Self::new_minimal).
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn par_new_minimal(items: Vec<I>, metric: M) -> Result<Self, &'static str>
    where
        I: Send + Sync,
        T: Send + Sync,
        M: Send + Sync,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let mut items = items.into_iter().enumerate().collect::<Vec<_>>();
        let root = Cluster::par_new_root(&mut items, &metric, &PartitionStrategy::default(), &|_, _, _| None);

        Ok(Self { items, root, metric })
    }
}

impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    /// Changes the metric used in the tree to the provided one.
    pub fn with_metric<N>(self, metric: N) -> Tree<Id, I, T, A, N>
    where
        N: Fn(&I, &I) -> T,
    {
        Tree {
            items: self.items,
            root: self.root,
            metric,
        }
    }

    /// Creates a new `Tree` from the given items and metric.
    ///
    /// # Arguments
    ///
    /// * `items` - A vector of tuples, each containing an identifier and an item.
    /// * `metric` - A function that computes the distance between two items.
    /// * `strategy` - A `PartitionStrategy` that defines how to partition clusters.
    /// * `post_process` - A function that computes auxiliary data for each cluster after partitioning, and may have side effects on the items. This will be
    ///   called for each cluster after it has been partitioned and its children have been assigned. It will receive a reference to the cluster, a mutable slice
    ///   of the items in the cluster, and a reference to the metric. It should return an `Option<A>` containing the auxiliary data for the cluster, or `None`
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn new<P, Post>(
        mut items: Vec<(Id, I)>,
        metric: M,
        strategy: &PartitionStrategy<P>,
        post_process: &Post,
    ) -> Result<Self, &'static str>
    where
        P: Fn(&Cluster<T, A>) -> bool,
        Post: Fn(&mut Cluster<T, A>, &mut [(Id, I)], &M) -> Option<A>,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let root = Cluster::new_root(&mut items, &metric, strategy, post_process);

        Ok(Self { items, root, metric })
    }

    /// Parallel version of [`new`](Self::new).
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn par_new<P, Post>(
        mut items: Vec<(Id, I)>,
        metric: M,
        strategy: &PartitionStrategy<P>,
        post_process: &Post,
    ) -> Result<Self, &'static str>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        M: Send + Sync,
        A: Send + Sync,
        P: Fn(&Cluster<T, A>) -> bool + Send + Sync,
        Post: Fn(&mut Cluster<T, A>, &mut [(Id, I)], &M) -> Option<A> + Send + Sync,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let root = Cluster::par_new_root(&mut items, &metric, strategy, post_process);

        Ok(Self { items, root, metric })
    }

    /// Returns the distance between the query item and the center of the given cluster.
    pub fn distance_to_center(&self, query: &I, cluster: &Cluster<T, A>) -> T {
        (self.metric)(query, self.center_of_cluster(cluster))
    }

    /// Returns the distances between the query item and all items in the given cluster, excluding the cluster's center.
    pub fn distances_to_items_in_subtree(&self, query: &I, cluster: &Cluster<T, A>) -> Vec<(usize, T)> {
        cluster
            .subtree_indices()
            .zip(self.items_in_subtree(cluster))
            .map(|(i, (_, item))| (i, (self.metric)(query, item)))
            .collect()
    }

    /// Returns the distances between the query item and all items in the given cluster, including the cluster's center.
    pub fn distances_to_items_in_cluster(&self, query: &I, cluster: &Cluster<T, A>) -> Vec<(usize, T)> {
        cluster
            .all_items_indices()
            .zip(self.items_in_cluster(cluster))
            .map(|(i, (_, item))| (i, (self.metric)(query, item)))
            .collect()
    }

    /// Parallel version of [`distances_to_items_in_subtree`](Self::distances_to_items_in_subtree).
    pub fn par_distances_to_items_in_subtree(&self, query: &I, cluster: &Cluster<T, A>) -> Vec<(usize, T)>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        cluster
            .subtree_indices()
            .into_par_iter()
            .zip(self.items_in_subtree(cluster))
            .map(|(i, (_, item))| (i, (self.metric)(query, item)))
            .collect()
    }

    /// Parallel version of [`distances_to_items_in_cluster`](Self::distances_to_items_in_cluster).
    pub fn par_distances_to_items_in_cluster(&self, query: &I, cluster: &Cluster<T, A>) -> Vec<(usize, T)>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        cluster
            .all_items_indices()
            .into_par_iter()
            .zip(self.items_in_cluster(cluster))
            .map(|(i, (_, item))| (i, (self.metric)(query, item)))
            .collect()
    }
}

impl<Id, I, T, A, M> Tree<Id, I, T, A, M> {
    /// Returns a reference to the identifier of the center item of the given cluster.
    pub fn center_id_of_cluster(&self, cluster: &Cluster<T, A>) -> &Id {
        &self.items[cluster.center_index()].0
    }

    /// Returns a reference to the center item of the given cluster.
    pub fn center_of_cluster(&self, cluster: &Cluster<T, A>) -> &I {
        &self.items[cluster.center_index()].1
    }

    /// Returns a slice of the items in the given cluster, excluding the cluster's center.
    pub fn items_in_subtree(&self, cluster: &Cluster<T, A>) -> &[(Id, I)] {
        &self.items[cluster.subtree_indices()]
    }

    /// Returns a slice of the items in the given cluster, including the cluster's center.
    pub(crate) fn items_in_cluster(&self, cluster: &Cluster<T, A>) -> &[(Id, I)] {
        &self.items[cluster.all_items_indices()]
    }

    /// Consumes the tree and returns all items stored in it.
    pub fn take_items(self) -> Vec<(Id, I)> {
        self.items
    }

    /// Returns a reference to the root cluster of the tree.
    pub const fn root(&self) -> &Cluster<T, A> {
        &self.root
    }

    /// Returns a mutable reference to the root cluster of the tree.
    pub const fn root_mut(&mut self) -> &mut Cluster<T, A> {
        &mut self.root
    }

    /// Returns the number of items stored in the tree.
    pub const fn cardinality(&self) -> usize {
        self.items.len()
    }

    /// Returns a reference to the metric used in the tree.
    pub const fn metric(&self) -> &M {
        &self.metric
    }

    /// Returns a vector of references to all clusters in the tree, in pre-order traversal.
    pub fn all_clusters_preorder(&self) -> Vec<&Cluster<T, A>> {
        self.root.subtree_preorder()
    }
}

#[cfg(feature = "serde")]
impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    Id: serde::Serialize + serde::de::DeserializeOwned,
    I: serde::Serialize + serde::de::DeserializeOwned,
    T: serde::Serialize + serde::de::DeserializeOwned,
    A: serde::Serialize + serde::de::DeserializeOwned,
{
    /// Serializes the `Tree` using Serde.
    ///
    /// # Errors
    ///
    /// If serialization fails.
    pub fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        /// A helper struct for serialization.
        #[derive(serde::Serialize)]
        struct SerializableTree<'a, Id, I, T, A> {
            /// The items stored in the tree, each paired with its identifier.
            items: &'a Vec<(Id, I)>,
            /// The root cluster of the tree.
            root: &'a Cluster<T, A>,
        }

        let serializable_tree = SerializableTree {
            items: &self.items,
            root: &self.root,
        };

        serializable_tree.serialize(serializer)
    }

    /// Deserializes a `Tree` using Serde.
    ///
    /// # Errors
    ///
    /// If deserialization fails.
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(deserializer: D, metric: M) -> Result<Self, D::Error> {
        /// A helper struct for deserialization.
        #[derive(serde::Deserialize)]
        struct SerializableTree<Id, I, T, A> {
            /// The items stored in the tree, each paired with its identifier.
            items: Vec<(Id, I)>,
            /// The root cluster of the tree.
            root: Cluster<T, A>,
        }

        let SerializableTree { items, root } = SerializableTree::deserialize(deserializer)?;
        Ok(Self { items, root, metric })
    }
}
#[cfg(feature = "serde")]
impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    Id: bitcode::Encode + bitcode::Decode,
    I: bitcode::Encode + bitcode::Decode,
    T: bitcode::Encode + bitcode::Decode,
    A: bitcode::Encode + bitcode::Decode,
{
    /// Encodes the `Tree` using Bitcode, returning a vector of bytes.
    ///
    /// # Errors
    ///
    /// If the encoding fails.
    pub fn bitcode_encode(&self) -> Result<Vec<u8>, bitcode::Error> {
        bitcode::encode(&(&self.items, &self.root))
    }

    /// Decodes a `Tree` using Bitcode, taking a slice of bytes and a metric.
    ///
    /// # Errors
    ///
    /// If the decoding fails.
    pub fn bitcode_decode(bytes: &[u8], metric: M) -> Result<Self, bitcode::Error> {
        let (items, root) = bitcode::decode(bytes)?;
        Ok(Self { items, root, metric })
    }
}
