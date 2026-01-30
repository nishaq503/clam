//! A `Tree` of `Clusters` for use in CLAM.

#![expect(clippy::type_complexity)]

use std::collections::HashMap;

use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::DistanceValue;

mod cluster;
mod partition;

pub use cluster::Cluster;
pub use partition::strategy::{self as partition_strategy, PartitionStrategy};

pub use cluster::AnnotatedItems;

/// The `Tree` struct is the main data structure used in CLAM.
///
/// If contains the root `Cluster`, the items stored in it, and the metric used to compute distances between items.
///
/// # Type Parameters
///
/// - `Id`: The type of the identifier for each item in the tree.
/// - `I`: The type of the items stored in the tree.
/// - `T`: The type of the distance values used in the tree.
/// - `A`: The type of any annotations that can be added to clusters.
/// - `M`: The type of the metric function used to compute distances between items.
#[must_use]
#[derive(Clone, Debug)]
pub struct Tree<Id, I, T, A, M> {
    /// The items stored in the tree, each paired with its identifier.
    pub(crate) items: Vec<(Id, I)>,
    /// The root cluster of the tree.
    pub(crate) root: Cluster<T, A>,
    /// All clusters in the tree. This is a mapping from `cluster.center_index` to `cluster`.
    pub(crate) cluster_map: HashMap<usize, Cluster<T, A>>,
    /// The metric used to compute distances between items.
    pub(crate) metric: M,
}

/// Minimal constructors for `Tree`.
///
/// - The identifier type is set to `usize` and will be the index of the item in the original vector.
/// - The annotation type is set to `()`, meaning that no annotations are stored in the tree.
/// - The default [`PartitionStrategy`](PartitionStrategy) is used to build a binary tree.
impl<I, T, M> Tree<usize, I, T, (), M>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    /// Creates a new `Tree` from the given items and metric.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn new_minimal(items: Vec<I>, metric: M) -> Result<Self, &'static str> {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let items = items.into_iter().enumerate().collect::<Vec<_>>();
        Self::new(items, metric, &PartitionStrategy::default(), &|_| (), 128)
    }

    /// Parallel version of [`Self::new_minimal`].
    ///
    /// # Errors
    ///
    /// See [`Self::new_minimal`].
    pub fn par_new_minimal(items: Vec<I>, metric: M) -> Result<Self, &'static str>
    where
        I: Send + Sync,
        T: Send + Sync,
        M: Send + Sync,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let items = items.into_iter().enumerate().collect::<Vec<_>>();
        Self::par_new(items, metric, &PartitionStrategy::default(), &|_| (), 128)
    }
}

/// Various getter methods for `Tree`.
impl<Id, I, T, A, M> Tree<Id, I, T, A, M> {
    /// Provides ownership of the members of the `Tree`.
    pub fn into_parts(self) -> (Vec<(Id, I)>, Cluster<T, A>, HashMap<usize, Cluster<T, A>>, M) {
        (self.items, self.root, self.cluster_map, self.metric)
    }

    /// Creates a `Tree` from its parts.
    pub const fn from_parts(items: Vec<(Id, I)>, root: Cluster<T, A>, cluster_map: HashMap<usize, Cluster<T, A>>, metric: M) -> Self {
        Self {
            items,
            root,
            cluster_map,
            metric,
        }
    }

    /// Returns a reference to the identifier of the center item of the given cluster.
    pub fn center_id_of_cluster(&self, cluster: &Cluster<T, A>) -> &Id {
        &self.items[cluster.center_index()].0
    }

    /// Returns a reference to the center item of the given cluster.
    pub fn center_of_cluster(&self, cluster: &Cluster<T, A>) -> &I {
        &self.items[cluster.center_index()].1
    }

    /// Returns a reference to all items in the tree.
    pub fn items(&self) -> &[(Id, I)] {
        &self.items
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

    /// Changes the metric used in the tree to the provided one.
    pub fn with_metric<N>(self, metric: N) -> Tree<Id, I, T, A, N> {
        Tree {
            items: self.items,
            root: self.root,
            cluster_map: self.cluster_map,
            metric,
        }
    }

    /// Returns a vector of references to all clusters in the tree, in pre-order traversal.
    pub fn all_clusters_postorder(&self) -> Vec<&Cluster<T, A>>
    where
        T: DistanceValue,
    {
        self.root.as_postorder_stack()
    }

    /// Returns a vector of references to all clusters in the tree that satisfy the given predicate.
    ///
    /// Once the predicate returns `true` for a cluster, its subtree is not searched further.
    pub fn filter_clusters<P, Args>(&self, predicate: P, args: &Args) -> Vec<&Cluster<T, A>>
    where
        P: Fn(&Cluster<T, A>, &Args) -> bool,
    {
        self.root.filter_clusters(&predicate, args)
    }

    /// Returns a vector of mutable references to all clusters in the tree that satisfy the given predicate.
    ///
    /// Once the predicate returns `true` for a cluster, its subtree is not searched further.
    pub fn filter_clusters_mut<P, Args>(&mut self, predicate: P, args: &Args) -> Vec<&mut Cluster<T, A>>
    where
        P: Fn(&Cluster<T, A>, &Args) -> bool,
    {
        self.root.filter_clusters_mut(&predicate, args)
    }

    /// Returns a vector of references to all leaf clusters in the tree.
    pub fn leaf_clusters(&self) -> Vec<&Cluster<T, A>> {
        self.filter_clusters(|cluster, ()| cluster.is_leaf(), &())
    }

    /// Returns a vector of mutable references to all leaf clusters in the tree.
    pub fn leaf_clusters_mut(&mut self) -> Vec<&mut Cluster<T, A>> {
        self.filter_clusters_mut(|cluster, ()| cluster.is_leaf(), &())
    }
}

/// Constructors and methods for computing distances in `Tree`.
impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    /// Creates a new `Tree` from the given items and metric.
    ///
    /// # Arguments
    ///
    /// * `items` - A vector of tuples, each containing an identifier and an item.
    /// * `metric` - A function that computes the distance between two items.
    /// * `strategy` - A `PartitionStrategy` that defines how to partition clusters.
    /// * `annotator` - A function that annotates clusters in post-order.
    /// * `max_recursive_depth` - The maximum depth of any recursion.
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn new<P, Ann>(
        mut items: Vec<(Id, I)>,
        metric: M,
        strategy: &PartitionStrategy<P>,
        annotator: &Ann,
        max_recursion_depth: usize,
    ) -> Result<Self, &'static str>
    where
        P: Fn(&Cluster<T, A>) -> bool,
        Ann: Fn(&Cluster<T, A>) -> A,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }
        ftlog::info!("Creating tree with {} items", items.len());

        let root = Cluster::new_root(&mut items, &metric, strategy, annotator, max_recursion_depth);
        let cluster_map = HashMap::new();

        ftlog::info!("Finished creating tree with {} items", items.len());
        Ok(Self {
            items,
            root,
            cluster_map,
            metric,
        })
    }

    /// Creates a new `Tree` from the given items and metric.
    ///
    /// # Arguments
    ///
    /// * `items` - A vector of tuples, each containing an identifier and an item.
    /// * `metric` - A function that computes the distance between two items.
    /// * `strategy` - A `PartitionStrategy` that defines how to partition clusters.
    /// * `annotator` - A function that annotates clusters in post-order.
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn new_iterative<P, Ann>(mut items: Vec<(Id, I)>, metric: M, strategy: &PartitionStrategy<P>, annotator: &Ann) -> Result<Self, &'static str>
    where
        P: Fn(&Cluster<T, A>) -> bool,
        Ann: Fn(&Cluster<T, A>) -> A,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }
        ftlog::info!("Creating tree with {} items", items.len());

        let mut cluster_map = HashMap::new();
        let mut stack = vec![Cluster::new_iterative(0, 0, &mut items, &metric, strategy)];
        while let Some((mut cluster, child_items)) = stack.pop() {
            stack.extend(
                child_items
                    .into_iter()
                    .map(|(child_center_index, c_items)| Cluster::new_iterative(cluster.depth() + 1, child_center_index, c_items, &metric, strategy))
                    .collect::<Vec<_>>(),
            );

            cluster.annotation = annotator(&cluster);
            cluster_map.insert(cluster.center_index(), cluster);
        }

        // TODO(Najib): Remove this pop after completing the iterative implementation.
        let root = cluster_map
            .remove(&0)
            .unwrap_or_else(|| unreachable!("Root cluster should be present in the cluster map."));

        ftlog::info!("Finished creating tree with {} items", items.len());
        Ok(Self {
            items,
            root,
            cluster_map,
            metric,
        })
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
}

/// Parallelized constructors and methods for computing distances in `Tree`.
impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    /// Parallel version of [`Self::new`].
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn par_new<P, Ann>(
        mut items: Vec<(Id, I)>,
        metric: M,
        strategy: &PartitionStrategy<P>,
        annotator: &Ann,
        max_recursion_depth: usize,
    ) -> Result<Self, &'static str>
    where
        P: Fn(&Cluster<T, A>) -> bool + Send + Sync,
        Ann: Fn(&Cluster<T, A>) -> A + Send + Sync,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }
        ftlog::info!("Creating tree with {} items in parallel", items.len());

        let root = Cluster::par_new_root(&mut items, &metric, strategy, annotator, max_recursion_depth);
        let cluster_map = HashMap::new();

        ftlog::info!("Finished creating tree with {} items in parallel", items.len());
        Ok(Self {
            items,
            root,
            cluster_map,
            metric,
        })
    }

    /// Parallel version of [`Self::new_iterative`].
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn par_new_iterative<P, Ann>(mut items: Vec<(Id, I)>, metric: M, strategy: &PartitionStrategy<P>, annotator: &Ann) -> Result<Self, &'static str>
    where
        P: Fn(&Cluster<T, A>) -> bool + Send + Sync,
        Ann: Fn(&Cluster<T, A>) -> A + Send + Sync,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }
        ftlog::info!("Creating tree with {} items", items.len());

        let mut cluster_map = HashMap::new();
        let mut stack = vec![Cluster::par_new_iterative(0, 0, &mut items, &metric, strategy)];
        while let Some((mut cluster, child_items)) = stack.pop() {
            stack.extend(
                child_items
                    .into_par_iter()
                    .map(|(child_center_index, c_items)| Cluster::par_new_iterative(cluster.depth() + 1, child_center_index, c_items, &metric, strategy))
                    .collect::<Vec<_>>(),
            );

            cluster.annotation = annotator(&cluster);
            cluster_map.insert(cluster.center_index(), cluster);
        }

        // TODO(Najib): Remove this pop after completing the iterative implementation.
        let root = cluster_map
            .remove(&0)
            .unwrap_or_else(|| unreachable!("Root cluster should be present in the cluster map."));

        ftlog::info!("Finished creating tree with {} items", items.len());
        Ok(Self {
            items,
            root,
            cluster_map,
            metric,
        })
    }

    /// Parallel version of [`distances_to_items_in_subtree`](Self::distances_to_items_in_subtree).
    pub fn par_distances_to_items_in_subtree(&self, query: &I, cluster: &Cluster<T, A>) -> Vec<(usize, T)> {
        cluster
            .subtree_indices()
            .into_par_iter()
            .zip(self.items_in_subtree(cluster))
            .map(|(i, (_, item))| (i, (self.metric)(query, item)))
            .collect()
    }

    /// Parallel version of [`distances_to_items_in_cluster`](Self::distances_to_items_in_cluster).
    pub fn par_distances_to_items_in_cluster(&self, query: &I, cluster: &Cluster<T, A>) -> Vec<(usize, T)> {
        cluster
            .all_items_indices()
            .into_par_iter()
            .zip(self.items_in_cluster(cluster))
            .map(|(i, (_, item))| (i, (self.metric)(query, item)))
            .collect()
    }
}

/// Serialization and deserialization methods for [`Tree`], gated by the `serde` feature.
///
/// These methods will only serialize and deserialize the items and the root cluster as a tuple. They will ignore the metric. This is because the metric is
/// typically a closure or function pointer, which cannot be serialized or deserialized. After deserialization, the metric must be provided using the
/// [`Tree::with_metric`] method.
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
        (&self.items, &self.root, &self.cluster_map).serialize(serializer)
    }

    /// Deserializes a `Tree` using Serde.
    ///
    /// # Errors
    ///
    /// If deserialization fails.
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(deserializer: D, metric: M) -> Result<Self, D::Error> {
        let (items, root, cluster_map) = <(_, _, _)>::deserialize(deserializer)?;
        Ok(Self {
            items,
            root,
            cluster_map,
            metric,
        })
    }
}

#[cfg(feature = "serde")]
impl<Id, I, T, A, M> databuf::Encode for Tree<Id, I, T, A, M>
where
    Id: databuf::Encode,
    I: databuf::Encode,
    T: databuf::Encode,
    A: databuf::Encode,
{
    fn encode<const CONFIG: u16>(&self, buffer: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        self.items.encode::<CONFIG>(buffer)?;
        self.root.encode::<CONFIG>(buffer)?;
        self.cluster_map.encode::<CONFIG>(buffer)
    }
}

#[cfg(feature = "serde")]
impl<'de, Id, I, T, A> databuf::Decode<'de> for Tree<Id, I, T, A, Box<dyn Fn(&I, &I) -> T>>
where
    Id: databuf::Decode<'de>,
    I: databuf::Decode<'de>,
    T: databuf::Decode<'de> + DistanceValue,
    A: databuf::Decode<'de>,
{
    fn decode<const CONFIG: u16>(buffer: &mut &'de [u8]) -> databuf::Result<Self> {
        let items = databuf::Decode::decode::<CONFIG>(buffer)?;
        let root = databuf::Decode::decode::<CONFIG>(buffer)?;
        let cluster_map = databuf::Decode::decode::<CONFIG>(buffer)?;
        let metric = Box::new(|_: &I, _: &I| T::zero()); // Placeholder; actual metric must be provided externally
        Ok(Self {
            items,
            root,
            cluster_map,
            metric,
        })
    }
}
