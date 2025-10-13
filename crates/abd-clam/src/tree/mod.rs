//! A `Tree` of `Clusters` for use in CLAM.

use crate::DistanceValue;

pub mod cakes;
mod node;
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
