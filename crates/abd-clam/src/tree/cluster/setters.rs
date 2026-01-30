//! Setters for `Cluster` properties.

use crate::DistanceValue;

use super::Cluster;

/// Setters for `Cluster` properties.
impl<T, A> Cluster<T, A> {
    /// Sets the depth of this cluster in the tree.
    pub const fn set_depth(&mut self, depth: usize) {
        self.depth = depth;
    }

    /// Sets the center item index.
    pub const fn set_center_index(&mut self, center_index: usize) {
        self.center_index = center_index;
    }

    /// Sets the cardinality of this cluster.
    pub const fn set_cardinality(&mut self, cardinality: usize) {
        self.cardinality = cardinality;
    }

    /// Sets the radius of this cluster.
    pub const fn set_radius(&mut self, radius: T)
    where
        T: DistanceValue,
    {
        self.radius = radius;
    }

    /// Sets the local fractal dimension of this cluster.
    pub const fn set_lfd(&mut self, lfd: f64) {
        self.lfd = lfd;
    }

    /// Sets the children and span of this cluster, replacing any existing children.
    pub fn set_children_and_span(&mut self, children: Box<[Self]>, child_center_indices: Box<[usize]>, span: T)
    where
        T: DistanceValue,
    {
        self.children = Some((children, child_center_indices, span));
    }

    /// Sets the annotation of this cluster, replacing any existing annotation.
    pub fn set_annotation(&mut self, annotation: A) {
        self.annotation = annotation;
    }

    /// Sets the annotation of this cluster using the given function and additional parameter, returning the old annotation.
    ///
    /// The function is called with:
    ///
    /// - A reference to this cluster.
    /// - A reference to the old annotation.
    /// - The additional parameter `b`, which may contain any extra information needed to compute the new annotation.
    pub fn annotate_with<F: FnOnce(&Self, &A, B) -> A, B>(&mut self, f: F, b: B) -> A {
        let new_annotation = f(self, &self.annotation, b);
        std::mem::replace(&mut self.annotation, new_annotation)
    }
}
