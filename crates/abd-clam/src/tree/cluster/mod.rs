//! A `Cluster` in a `Tree` for use in CLAM.

mod par_partition;
mod partition;
mod partition_strategy;
mod to_csv;

pub use partition::{lfd_estimate, reorder_items_in_place};
pub use partition_strategy::{BranchingFactor, PartitionStrategy, SpanReductionFactor};

/// A cluster in the `Tree`.
pub struct Cluster<T, A> {
    /// Depth of this cluster in the tree, with root at depth 0.
    pub(crate) depth: usize,
    /// Index of the center item in the `items` array of the `Tree`.
    pub(crate) center_index: usize,
    /// Number of items in the subtree rooted at this cluster, including the center item.
    pub(crate) cardinality: usize,
    /// The distance from the center item to the furthest item in the subtree.
    pub(crate) radius: T,
    /// The Local Fractal Dimension of the `Cluster`.
    pub(crate) lfd: f64,
    /// The children and span of this cluster, if it was partitioned. The span is the distance between the two poles used to partition the cluster.
    pub(crate) children: Option<(Box<[Self]>, T)>,
    /// Optional arbitrary data associated with this cluster.
    pub(crate) annotation: Option<A>,
}

impl<T, A> core::fmt::Display for Cluster<T, A>
where
    T: core::fmt::Display,
    A: core::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut fields = vec![
            format!("depth: {}", self.depth),
            format!("center_index: {}", self.center_index),
            format!("cardinality: {}", self.cardinality),
            format!("radius: {}", self.radius),
            format!("lfd: {:.6}", self.lfd),
            format!(
                "indices: {}..{}",
                self.center_index + 1,
                self.center_index + self.cardinality
            ),
        ];
        if let Some(annotation) = &self.annotation {
            fields.push(format!("annotation: {:?}", annotation));
        }
        if let Some((children, span)) = &self.children {
            fields.push(format!("span: {}", span));
            fields.push(format!("children: {}", children.len()));
        }

        let joined = fields.join(", ");
        write!(f, "Cluster({})", joined)
    }
}

impl<T, A> deepsize::DeepSizeOf for Cluster<T, A>
where
    T: deepsize::DeepSizeOf,
    A: deepsize::DeepSizeOf,
{
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        core::mem::size_of::<usize>()  // for self.depth
            + core::mem::size_of::<usize>()  // for self.center_index
            + core::mem::size_of::<usize>()  // for self.cardinality
            + self.radius.deep_size_of_children(context)
            + core::mem::size_of::<f64>()  // for self.lfd
            + self.children.as_ref().map_or(0, |(children, span)| {
                span.deep_size_of_children(context)
                    + children.iter().map(|child| child.deep_size_of_children(context)).sum::<usize>()
            })
            + self.annotation.deep_size_of_children(context)
    }
}

impl<T, A> Cluster<T, A> {
    /// Returns the depth of this cluster in the tree.
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the index of the center item in the `items` array of the `Tree`.
    pub const fn center_index(&self) -> usize {
        self.center_index
    }

    /// Returns the number of items in the subtree rooted at this cluster, including the center item.
    pub const fn cardinality(&self) -> usize {
        self.cardinality
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

    /// Returns the distance from the center item to the furthest item in the subtree.
    pub const fn radius(&self) -> T
    where
        T: Copy,
    {
        self.radius
    }

    /// Returns the Local Fractal Dimension (LFD) of this cluster.
    pub const fn lfd(&self) -> f64 {
        self.lfd
    }

    /// Returns true if this cluster is a singleton (i.e., contains exactly one item or has a radius of zero).
    pub fn is_singleton(&self) -> bool
    where
        T: num_traits::Zero,
    {
        self.cardinality == 1 || self.radius.is_zero()
    }

    /// Returns true if this cluster is a leaf (i.e., has no children).
    pub const fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// Returns a reference to the children of this cluster, if any.
    ///
    /// Use this to read the subtree rooted at this cluster without taking ownership.
    pub fn children(&self) -> Option<&[Self]> {
        self.children.as_ref().map(|(children, _)| children.as_ref())
    }

    /// Returns the span of this cluster, if it has children.
    ///
    /// The span is the distance between the two poles used to partition the cluster.
    pub fn span(&self) -> Option<&T> {
        self.children.as_ref().map(|(_, span)| span)
    }

    /// Returns all clusters in the subtree rooted at this cluster, including this cluster, in pre-order.
    pub fn subtree_preorder(&self) -> Vec<&Self> {
        if let Some((children, _)) = &self.children {
            core::iter::once(self)
                .chain(children.iter().flat_map(|child| child.subtree_preorder()))
                .collect()
        } else {
            vec![self]
        }
    }

    /// Returns true if this cluster has an annotation.
    pub const fn is_annotated(&self) -> bool {
        self.annotation.is_some()
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

    /// Annotates the cluster with the given value, replacing any existing annotation.
    pub fn annotate(&mut self, annotation: A) {
        self.annotation = Some(annotation);
    }

    /// Annotates the cluster by evaluating the given closure, returning any existing annotation.
    ///
    /// The closure is called with a reference to the cluster and a reference to the previous annotation (if any).
    pub fn annotate_with<F: FnOnce(&Self, Option<&A>) -> A>(&mut self, f: F) -> Option<A> {
        let old_annotation = self.annotation.take();
        self.annotation = Some(f(self, old_annotation.as_ref()));
        old_annotation
    }

    /// Changes the annotations, and their types, of this cluster and all its descendants by applying the given closure recursively in pre-order.
    ///
    /// The closure is called with a reference to each cluster before its children are processed and its return value becomes the new annotation for that cluster.
    pub fn annotate_pre_order<B, F: FnMut(&Self) -> Option<B>>(mut self, f: &mut F) -> Cluster<T, B> {
        let annotation = f(&self);

        let children = self.children.take().map(|(boxed_children, span)| {
            let new_children = boxed_children
                .into_vec()
                .into_iter()
                .map(|child| child.annotate_pre_order(f))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            (new_children, span)
        });

        Cluster {
            depth: self.depth,
            center_index: self.center_index,
            cardinality: self.cardinality,
            radius: self.radius,
            lfd: self.lfd,
            children,
            annotation,
        }
    }

    /// Changes the annotations, and their types, of this cluster and all its descendants by applying the given closure recursively in post-order.
    ///
    /// The closure is called with a reference to each cluster after its children have been processed, along with the previous annotation (if any). The closure's
    /// return value becomes the new annotation for that cluster.
    pub fn annotate_post_order<B, F: FnMut(&Cluster<T, B>, Option<A>) -> Option<B>>(
        mut self,
        f: &mut F,
    ) -> Cluster<T, B> {
        let old_annotation = self.annotation.take();

        let children = self.children.take().map(|(boxed_children, span)| {
            let new_children = boxed_children
                .into_vec()
                .into_iter()
                .map(|child| child.annotate_post_order(f))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            (new_children, span)
        });

        let mut cluster = Cluster {
            depth: self.depth,
            center_index: self.center_index,
            cardinality: self.cardinality,
            radius: self.radius,
            lfd: self.lfd,
            children,
            annotation: None,
        };

        cluster.annotation = f(&cluster, old_annotation);

        cluster
    }
}
