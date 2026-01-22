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
    pub fn set_children_and_span(&mut self, children: Box<[Self]>, span: T)
    where
        T: DistanceValue,
    {
        self.children = Some((children, span));
    }

    /// Sets the annotation of this cluster, replacing any existing annotation.
    pub fn set_annotation(&mut self, annotation: A) {
        self.annotation = Some(annotation);
    }

    /// Sets the annotation of this cluster using the given function and additional parameter, returning the old annotation if any.
    ///
    /// The function is called with:
    ///
    /// - A reference to this cluster.
    /// - A reference to the old annotation, if any.
    /// - The additional parameter `b`, which may contain any extra information needed to compute the new annotation.
    pub fn annotate_with<F: FnOnce(&Self, Option<&A>, B) -> A, B>(&mut self, f: F, b: B) -> Option<A> {
        let old_annotation = self.annotation.take();
        self.annotation = Some(f(self, old_annotation.as_ref(), b));
        old_annotation
    }
}
