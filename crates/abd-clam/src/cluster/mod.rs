//! The `Cluster` struct provides the core clustering algorithm and data structure for CLAM.

use core::fmt::Debug;

pub use rayon::prelude::*;

use crate::DistanceValue;

pub mod annotations;
pub mod partition;

/// A `Cluster` is a collection of items in a dataset that are within a `radius` from a `center` item.
///
/// # Type Parameters
///
/// * `I`: The type of items in the tree.
/// * `Id`: The type of metadata associated each item.
/// * `T`: The type of distance values between items.
/// * `A`: The type of arbitrary annotations associated with each cluster.
#[must_use]
pub struct Cluster<Id, I, T: DistanceValue, A> {
    /// The number of items in the cluster, including the center.
    pub(crate) cardinality: usize,
    /// The center item of the cluster.
    pub(crate) center: (Id, I),
    /// The radius of the cluster.
    pub(crate) radius: T,
    /// The Local Fractal Dimension (LFD) of the cluster.
    pub(crate) lfd: f64,
    /// The sum of all radial distances from the center to all items in the cluster.
    pub(crate) radial_sum: T,
    /// The `Contents` of the cluster.
    pub(crate) contents: Contents<Id, I, T, A>,
    /// Arbitrary annotation for the cluster.
    pub(crate) annotation: Option<A>,
}

/// The contents of a `Cluster` can either be a collection of items (if it is a leaf) or a collection of child `Cluster`s (if it is a parent).
pub(crate) enum Contents<Id, I, T: DistanceValue, A> {
    /// The cluster is a leaf and contains items directly.
    Leaf(Vec<(Id, I)>),
    /// The cluster is a parent and contains child clusters.
    Children([Box<Cluster<Id, I, T, A>>; 2]),
}

impl<I: Debug, Id: Debug, T: DistanceValue + Debug, A: Debug> Debug for Cluster<Id, I, T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cluster")
            .field("cardinality", &self.cardinality)
            .field("center", &self.center)
            .field("radius", &self.radius)
            .field("lfd", &self.lfd)
            .field("radial_sum", &self.radial_sum)
            .field("contents", &self.contents)
            .field("annotation", &self.annotation)
            .finish()
    }
}

impl<Id, I, T: DistanceValue + Debug, A: Debug> Debug for Contents<Id, I, T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Leaf(items) => f.debug_tuple("Leaf").field(&items.len()).finish(),
            Self::Children(_) => f.debug_tuple("Children").finish(),
        }
    }
}

impl<I, T: DistanceValue> Cluster<usize, I, T, ()> {
    /// Create a new tree of `Cluster`s with `usize` indices as item metadata, and no annotations.
    ///
    /// # Errors
    ///
    /// - See [`new_tree`](Self::new_tree) for details.
    pub fn new_tree_minimal<M: Fn(&I, &I) -> T>(
        items: Vec<I>,
        metric: &M,
        criteria: &impl Fn(&Self) -> bool,
    ) -> Result<Self, String> {
        let indexed_items = items.into_iter().enumerate().collect();
        Self::new_tree(indexed_items, metric, criteria)
    }
}

impl<I: Send + Sync, T: DistanceValue + Send + Sync> Cluster<usize, I, T, ()> {
    /// Parallel version of [`new_tree_minimal`](Self::new_tree_minimal).
    ///
    /// # Errors
    ///
    /// - See [`new_tree`](Self::new_tree) for details.
    pub fn par_new_tree_minimal<M: Fn(&I, &I) -> T + Send + Sync>(
        items: Vec<I>,
        metric: &M,
        criteria: &(impl Fn(&Self) -> bool + Send + Sync),
    ) -> Result<Self, String> {
        let indexed_items = items.into_iter().enumerate().collect();
        Self::par_new_tree(indexed_items, metric, criteria)
    }
}

impl<Id, I, T: DistanceValue, A> Cluster<Id, I, T, A> {
    /// The number of items in the cluster, including the center.
    pub const fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Returns a reference to the id of the center item of the cluster.
    pub const fn center_id(&self) -> &Id {
        &self.center.0
    }

    /// A reference to the center item of the cluster.
    pub const fn center(&self) -> &I {
        &self.center.1
    }

    /// The radius of the cluster.
    pub const fn radius(&self) -> T {
        self.radius
    }

    /// The Local Fractal Dimension (LFD) of the cluster.
    pub const fn lfd(&self) -> f64 {
        self.lfd
    }

    /// The sum of all radial distances from the center to all items in the cluster.
    pub const fn radial_sum(&self) -> T {
        self.radial_sum
    }

    /// A reference to the annotations, if any.
    pub const fn annotation(&self) -> Option<&A> {
        self.annotation.as_ref()
    }

    /// Checks if the cluster is a leaf.
    pub const fn is_leaf(&self) -> bool {
        matches!(self.contents, Contents::Leaf(_))
    }

    /// Checks if the cluster is a singleton (i.e., contains only one distinct item).
    pub fn is_singleton(&self) -> bool {
        self.cardinality == 1 || self.radius.is_zero()
    }

    /// The children of the cluster, if any.
    pub fn children(&self) -> Option<[&Self; 2]> {
        match &self.contents {
            Contents::Leaf(_) => None,
            Contents::Children([left, right]) => Some([left, right]),
        }
    }

    /// A vector of references to all clusters in the tree in pre-order (i.e., parent before children).
    pub fn subtree(&self) -> Vec<&Self> {
        match &self.contents {
            Contents::Leaf(_) => vec![self],
            Contents::Children([left, right]) => core::iter::once(self)
                .chain(left.subtree())
                .chain(right.subtree())
                .collect(),
        }
    }

    /// A vector of references to all items in the subtree rooted at this cluster,
    /// excluding the center of this cluster.
    pub fn subtree_items(&self) -> Vec<&(Id, I)> {
        match &self.contents {
            Contents::Leaf(items) => items.iter().collect(),
            Contents::Children([left, right]) => left.all_items().into_iter().chain(right.all_items()).collect(),
        }
    }

    /// A vector of references to all items in the cluster, including the center, which is placed first.
    pub fn all_items(&self) -> Vec<&(Id, I)> {
        match &self.contents {
            Contents::Leaf(items) => core::iter::once(&self.center).chain(items.iter()).collect(),
            Contents::Children([left, right]) => core::iter::once(&self.center)
                .chain(left.all_items())
                .chain(right.all_items())
                .collect(),
        }
    }

    /// Returns the distance from the given item to the center of the cluster using the provided metric.
    pub fn distance_to_center<M: Fn(&I, &I) -> T>(&self, item: &I, metric: &M) -> (&(Id, I), T) {
        (&self.center, metric(item, &self.center.1))
    }

    /// Returns the distance from the given item to all items in the cluster and its subtree using the provided metric.
    pub fn distances_to_all_items<M: Fn(&I, &I) -> T>(&self, item: &I, metric: &M) -> Vec<(&Id, &I, T)> {
        self.all_items().iter().map(|(i, p)| (i, p, metric(item, p))).collect()
    }

    /// Traverses the tree in pre-order, checking the provided predicate on each cluster, and converts clusters that satisfy the predicate into leaves by collecting
    /// all items from their descendants and dropping the descendants in the process.
    pub fn prune<P: Fn(&Self) -> bool>(&mut self, predicate: &P) {
        if predicate(self) {
            // The predicate is satisfied, so we convert this cluster to a leaf by collecting all items from its descendants.
            self.contents = Contents::Leaf(self.take_subtree_items());
        } else if let Contents::Children([left, right]) = &mut self.contents {
            // The predicate is not satisfied, so we continue checking children.
            left.prune(predicate);
            right.prune(predicate);
        }
    }
}

impl<Id: Send + Sync, I: Send + Sync, T: DistanceValue + Send + Sync, A: Send + Sync> Cluster<Id, I, T, A> {
    /// Parallel version of [`distance_to_all`](Self::distances_to_all).
    pub fn par_distances_to_all_items<M: Fn(&I, &I) -> T + Send + Sync>(
        &self,
        item: &I,
        metric: &M,
    ) -> Vec<(&Id, &I, T)> {
        self.all_items()
            .par_iter()
            .map(|(id, p)| (id, p, metric(item, p)))
            .collect()
    }

    /// Parallel version of [`prune`](Self::prune).
    pub fn par_prune<P: (Fn(&Self) -> bool) + Send + Sync>(&mut self, predicate: &P) {
        if predicate(self) {
            // The predicate is satisfied, so we convert this cluster to a leaf by collecting all items from its subtree.
            self.contents = Contents::Leaf(self.take_subtree_items());
        } else if let Contents::Children([left, right]) = &mut self.contents {
            // The predicate is not satisfied, so we continue checking children.
            rayon::join(|| left.par_prune(predicate), || right.par_prune(predicate));
        }
    }
}
