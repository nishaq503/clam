//! Getters for `Cluster` properties.

use crate::DistanceValue;

use super::Cluster;

/// Getters for `Cluster` properties that are stored as fields.
impl<T, A> Cluster<T, A> {
    /// Returns the depth of this cluster in the tree.
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Gets the center item index.
    pub const fn center_index(&self) -> usize {
        self.center_index
    }

    /// Gets the cardinality of this cluster.
    pub const fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Gets the radius of this cluster.
    pub const fn radius(&self) -> T
    where
        T: DistanceValue,
    {
        self.radius
    }

    /// Gets the local fractal dimension of this cluster.
    pub const fn lfd(&self) -> f64 {
        self.lfd
    }

    /// Gets the children of this cluster, if it was partitioned.
    pub fn children_and_span(&self) -> Option<(&[Self], &T)> {
        self.children.as_ref().map(|(children, span)| (children.as_ref(), span))
    }

    /// Gets the children of this cluster, if it was partitioned.
    pub fn children(&self) -> Option<&[Self]> {
        self.children.as_ref().map(|(children, _)| children.as_ref())
    }

    /// Returns a mutable reference to the children of this cluster, if any.
    pub fn children_mut(&mut self) -> Option<&mut [Self]> {
        self.children.as_mut().map(|(children, _)| children.as_mut())
    }

    /// Gets the span of this cluster, if it was partitioned.
    pub fn span(&self) -> Option<T>
    where
        T: DistanceValue,
    {
        self.children.as_ref().map(|(_, span)| *span)
    }

    /// Takes ownership of the children and span of this cluster, if any, leaving it a leaf cluster.
    pub const fn take_children_and_span(&mut self) -> Option<(Box<[Self]>, T)> {
        self.children.take()
    }

    /// Returns an optional reference to the cluster's annotation, if any.
    ///
    /// Use this to read metadata attached to the cluster without taking ownership.
    pub const fn annotation(&self) -> Option<&A> {
        self.annotation.as_ref()
    }

    /// Returns an optional mutable reference to the cluster's annotation, if any.
    ///
    /// Use this to modify cluster metadata in place.
    pub const fn annotation_mut(&mut self) -> Option<&mut A> {
        self.annotation.as_mut()
    }

    /// Removes and returns the cluster's annotation, if any, leaving it unannotated.
    pub const fn take_annotation(&mut self) -> Option<A> {
        self.annotation.take()
    }
}

/// Getters for `Cluster` properties that are cheaply computed from fields.
impl<T, A> Cluster<T, A> {
    /// Returns true if this cluster is a singleton (i.e., contains exactly one item or has a radius of zero).
    pub fn is_singleton(&self) -> bool
    where
        T: DistanceValue,
    {
        self.cardinality == 1 || self.radius.is_zero()
    }

    /// Returns true if this cluster is a leaf (i.e., has no children).
    pub const fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// Returns a `Range` that can be used to index into the `items` array of the `Tree` for all items in this cluster, including the center item.
    pub const fn all_items_indices(&self) -> std::ops::Range<usize> {
        self.center_index..(self.center_index + self.cardinality)
    }

    /// Returns a `Range` that can be used to index into the `items` array of the `Tree` for all items in the subtree rooted at this cluster, excluding the
    /// center item of this cluster.
    pub const fn subtree_indices(&self) -> std::ops::Range<usize> {
        (self.center_index + 1)..(self.center_index + self.cardinality)
    }

    /// Returns true if this cluster has an annotation.
    pub const fn is_annotated(&self) -> bool {
        self.annotation.is_some()
    }
}
