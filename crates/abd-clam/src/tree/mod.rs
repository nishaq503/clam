//! A `Tree` of `Clusters` for use in CLAM.

use crate::DistanceValue;
use rand::prelude::*;

mod node;

pub use node::{lfd_estimate, BranchingFactor, Node, PartitionStrategy, SpanReductionFactor};

/// A tree structure used in CLAM for organizing items based on a given metric.
pub struct Tree<Id, I, T, A, M> {
    /// The items stored in the tree, each paired with its identifier.
    items: Vec<(Id, I)>,
    /// The root node of the tree.
    root: Node<T, A>,
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
        let root = Node::new_root(&mut items, &metric, &PartitionStrategy::default());

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
        let root = Node::par_new_root(&mut items, &metric, &PartitionStrategy::default());

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
        P: Fn(&Node<T, A>) -> bool,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let root = Node::new_root(&mut items, &metric, strategy);

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
        P: Fn(&Node<T, A>) -> bool + Send + Sync,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let root = Node::par_new_root(&mut items, &metric, strategy);

        Ok(Self { items, root, metric })
    }

    /// Returns a reference to the items stored in the tree.
    pub fn items(&self) -> &[(Id, I)] {
        &self.items
    }

    /// Consumes the tree and returns all items stored in it.
    pub fn take_items(self) -> Vec<(Id, I)> {
        self.items
    }

    /// Returns a reference to the root node of the tree.
    pub const fn root(&self) -> &Node<T, A> {
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

    /// Returns a vector of references to all nodes in the tree, in pre-order traversal.
    pub fn all_nodes_preorder(&self) -> Vec<&Node<T, A>> {
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
