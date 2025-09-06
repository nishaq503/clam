//! Setters for `Cluster` properties.

use super::Cluster;

/// Setters for `Cluster` properties.
impl<T, A> Cluster<T, A> {
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
    /// - The additional parameter `args`, which may contain any extra information needed to compute the new annotation.
    pub fn annotate_with<F: FnOnce(&Self, &A, Args) -> A, Args>(&mut self, f: F, args: Args) -> A {
        let new_annotation = f(self, &self.annotation, args);
        std::mem::replace(&mut self.annotation, new_annotation)
    }
}
