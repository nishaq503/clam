//! A `Tree` of `Clusters` for use in CLAM.

use crate::DistanceValue;

pub mod cakes;
mod node;
mod par_partition;
mod partition;

pub use node::Node;

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
        // for self.metric
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
    pub fn new(mut items: Vec<(Id, I)>, metric: M) -> Result<Self, &'static str> {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let root = Node::new_root(&mut items, &metric);

        Ok(Self { items, root, metric })
    }

    /// Parallel version of [`new`](Self::new).
    ///
    /// # Errors
    ///
    /// If `items` is empty.
    pub fn par_new(mut items: Vec<(Id, I)>, metric: M) -> Result<Self, &'static str>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        M: Send + Sync,
        A: Send + Sync,
    {
        if items.is_empty() {
            return Err("Cannot create a Tree with no items.");
        }

        let root = Node::par_new_root(&mut items, &metric);

        Ok(Self { items, root, metric })
    }

    /// Returns a reference to the items stored in the tree.
    pub fn items(&self) -> &[(Id, I)] {
        &self.items
    }

    /// Returns a reference to the root node of the tree.
    pub const fn root(&self) -> &Node<T, A> {
        &self.root
    }

    /// Returns a reference to the metric used in the tree.
    pub const fn metric(&self) -> &M {
        &self.metric
    }
}
