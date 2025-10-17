//! A `Tree` of `Clusters` for use in CLAM.

use rand::prelude::*;
use rayon::prelude::*;

use crate::DistanceValue;

mod cluster;

pub use cluster::{lfd_estimate, BranchingFactor, Cluster, PartitionStrategy, SpanReductionFactor};

/// A tree structure used in CLAM for organizing items based on a given metric.
pub struct Tree<Id, I, T, A, M> {
    /// The items stored in the tree, each paired with its identifier.
    items: Vec<(Id, I)>,
    /// The root cluster of the tree.
    root: Cluster<T, A>,
    /// The metric used to compute distances between items.
    metric: M,
}

impl<Id, I, T, A, M> deepsize::DeepSizeOf for Tree<Id, I, T, A, M>
where
    Id: deepsize::DeepSizeOf,
    I: deepsize::DeepSizeOf,
    T: deepsize::DeepSizeOf,
    A: deepsize::DeepSizeOf,
{
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.items.deep_size_of_children(context) + self.root.deep_size_of_children(context) + core::mem::size_of::<M>()
    }
}

impl<I, T, M> Tree<usize, I, T, (), M>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    /// Creates a new `Tree` from the given items and metric.
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn new_minimal(items: Vec<I>, metric: M) -> Result<Self, &'static str>
    where
        I: core::fmt::Debug,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let mut items = items.into_iter().enumerate().collect::<Vec<_>>();
        let root = Cluster::new_root(&mut items, &metric, &PartitionStrategy::default());

        Ok(Self { items, root, metric })
    }

    /// Parallel version of [`new`](Self::new).
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn par_new_minimal(items: Vec<I>, metric: M) -> Result<Self, &'static str>
    where
        I: Send + Sync + core::fmt::Debug,
        T: Send + Sync,
        M: Send + Sync,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let mut items = items.into_iter().enumerate().collect::<Vec<_>>();
        let root = Cluster::par_new_root(&mut items, &metric, &PartitionStrategy::default());

        Ok(Self { items, root, metric })
    }
}

impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    /// Creates a new `Tree` from the given items and metric.
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn new<P>(mut items: Vec<(Id, I)>, metric: M, strategy: &PartitionStrategy<P>) -> Result<Self, &'static str>
    where
        Id: core::fmt::Debug,
        I: core::fmt::Debug,
        A: core::fmt::Debug,
        P: Fn(&Cluster<T, A>) -> bool,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let root = Cluster::new_root(&mut items, &metric, strategy);

        Ok(Self { items, root, metric })
    }

    /// Parallel version of [`new`](Self::new).
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn par_new<P>(mut items: Vec<(Id, I)>, metric: M, strategy: &PartitionStrategy<P>) -> Result<Self, &'static str>
    where
        Id: Send + Sync + core::fmt::Debug,
        I: Send + Sync + core::fmt::Debug,
        T: Send + Sync,
        M: Send + Sync,
        A: Send + Sync,
        P: Fn(&Cluster<T, A>) -> bool + Send + Sync,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let root = Cluster::par_new_root(&mut items, &metric, strategy);

        Ok(Self { items, root, metric })
    }

    /// Returns a reference to the identifier of the center item of the given cluster.
    pub fn center_id_of_cluster(&self, cluster: &Cluster<T, A>) -> &Id {
        // SAFETY: cluster.center_index() is always a valid index into self.items
        #[allow(unsafe_code)]
        unsafe {
            &self.items.get_unchecked(cluster.center_index()).0
        }
    }

    /// Returns a reference to the center item of the given cluster.
    pub fn center_of_cluster(&self, cluster: &Cluster<T, A>) -> &I {
        // SAFETY: cluster.center_index() is always a valid index into self.items
        #[allow(unsafe_code)]
        unsafe {
            &self.items.get_unchecked(cluster.center_index()).1
        }
    }

    /// Returns a slice of the items in the given cluster, excluding the cluster's center.
    pub fn items_in_subtree(&self, cluster: &Cluster<T, A>) -> &[(Id, I)] {
        // SAFETY: cluster.subtree_indices() are always valid indices into self.items
        #[allow(unsafe_code)]
        unsafe {
            self.items.get_unchecked(cluster.subtree_indices())
        }
    }

    /// Returns a slice of the items in the given cluster, including the cluster's center.
    pub fn items_in_cluster(&self, cluster: &Cluster<T, A>) -> &[(Id, I)] {
        // SAFETY: cluster.all_items_indices() are always valid indices into self.items
        #[allow(unsafe_code)]
        unsafe {
            self.items.get_unchecked(cluster.all_items_indices())
        }
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

    /// Consumes the tree and returns all items stored in it.
    pub fn take_items(self) -> Vec<(Id, I)> {
        self.items
    }

    /// Returns a reference to the root cluster of the tree.
    pub const fn root(&self) -> &Cluster<T, A> {
        &self.root
    }

    /// Returns a reference to the metric used in the tree.
    pub const fn metric(&self) -> &M {
        &self.metric
    }

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

    /// Returns the number of items stored in the tree.
    pub const fn cardinality(&self) -> usize {
        self.items.len()
    }

    /// Returns a vector of references to all clusters in the tree, in pre-order traversal.
    pub fn all_clusters_preorder(&self) -> Vec<&Cluster<T, A>> {
        self.root.subtree_preorder()
    }

    /// Clones and returns a random subset of `n` items from the tree.
    ///
    /// If `n` is greater than the number of items in the tree, all items are returned.
    ///
    /// The order of items in the returned vector is random.
    pub fn random_subset<R: rand::Rng>(&self, n: usize, rng: &mut R) -> Vec<&I> {
        let n = n.min(self.items.len());
        let mut indices = (0..self.items.len()).collect::<Vec<_>>();
        indices.shuffle(rng);
        indices.truncate(n);
        indices.iter().map(|&i| &self.items[i].1).collect()
    }
}
