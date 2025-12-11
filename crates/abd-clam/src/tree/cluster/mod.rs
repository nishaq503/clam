//! A `Cluster` in a `Tree` for use in CLAM.

mod par_partition;
mod partition;
mod partition_strategy;
#[cfg(feature = "serde")]
mod to_csv;

pub use partition::{lfd_estimate, reorder_items_in_place};
pub use partition_strategy::{BranchingFactor, PartitionStrategy, SpanReductionFactor};

/// A cluster in the `Tree`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", expect(clippy::unsafe_derive_deserialize))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
#[must_use]
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

impl<T, A> PartialEq for Cluster<T, A>
where
    T: PartialEq,
    A: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth
            && self.center_index == other.center_index
            && self.cardinality == other.cardinality
            && self.radius == other.radius
            && self.lfd == other.lfd
            && self.children == other.children
            && self.annotation == other.annotation
    }
}

impl<T, A> core::fmt::Display for Cluster<T, A>
where
    T: core::fmt::Display,
    A: core::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut fields = vec![
            format!("d: {}", self.depth),
            format!("c: {}", self.center_index),
            format!("car: {}", self.cardinality),
            format!("r: {}", self.radius),
            format!("LFD: {:.3}", self.lfd),
        ];
        if self.cardinality == 1 {
        } else if self.cardinality == 2 {
            fields.push(format!("non center: {}", self.center_index + 1));
        } else {
            fields.push(format!("indices: {}..{}", self.center_index + 1, self.center_index + self.cardinality));
        }

        if let Some(annotation) = &self.annotation {
            fields.push(format!("annotation: {annotation:?}"));
        }
        let name = if let Some((children, span)) = &self.children {
            fields.push(format!("span: {span}"));

            let indented_children = children
                .iter()
                .map(|child| format!("|--{}", format!("{child}").replace('\n', "\n|  ")))
                .collect::<Vec<_>>();
            let children = indented_children.join("\n");
            fields.push(format!("\n{children}"));

            "P"
        } else {
            "L"
        };

        write!(f, "{name}: {}", fields.join(", "))
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

    /// Returns references to all clusters along the path from this cluster to the descendant that directly contains the item at the given item index.
    ///
    /// # Errors
    ///
    /// If the given item index is not contained in this cluster.
    pub fn path_to_cluster_containing(&self, item_index: usize) -> Result<Vec<&Self>, &'static str> {
        if item_index == self.center_index {
            Ok(vec![self])
        } else if self.subtree_indices().contains(&item_index) {
            let mut path = self.descend_to_cluster(item_index);
            path.reverse();
            Ok(path)
        } else {
            Err("item index out of bounds for this cluster")
        }
    }

    /// Helper function for [`Self::path_to_cluster_containing`].
    ///
    /// This returns references the path in reverse order (from descendant to root).
    fn descend_to_cluster(&self, item_index: usize) -> Vec<&Self> {
        if self.center_index == item_index {
            // The requested item is the center of this cluster.
            return vec![self];
        }

        if let Some((children, _)) = &self.children {
            let target_child = children
                .iter()
                .find(|child| child.all_items_indices().contains(&item_index))
                .unwrap_or_else(|| unreachable!("The item index is guaranteed to be in one of the children."));
            let mut path = target_child.descend_to_cluster(item_index);
            path.push(self);
            path
        } else {
            // This is a leaf cluster and the requested item is not the center.
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
    pub fn annotate_pre_order<B, F>(mut self, f: &F) -> Cluster<T, B>
    where
        F: Fn(&mut Self) -> Option<B>,
    {
        let annotation = f(&mut self);

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
    /// The closure is called with a reference to each cluster after its children have been processed, along with the previous annotation (if any). The
    /// closure's return value becomes the new annotation for that cluster.
    pub fn annotate_post_order<B, F>(mut self, f: &F) -> Cluster<T, B>
    where
        F: Fn(&mut Cluster<T, B>, Option<A>) -> B,
    {
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

        cluster.annotation = Some(f(&mut cluster, old_annotation));

        cluster
    }

    /// Clears the annotation of this cluster and all its descendants.
    pub fn clear_annotations(self) -> Cluster<T, ()> {
        let children = self.children.map(|(boxed_children, span)| {
            let new_children = boxed_children
                .into_vec()
                .into_iter()
                .map(Self::clear_annotations)
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
            annotation: None,
        }
    }

    /// Returns references to the clusters in the subtree rooted here that satisfy the given predicate.
    ///
    /// Once the predicate returns `true` for a cluster, its subtree is not searched further.
    pub fn select_clusters<P>(&self, predicate: &P) -> Vec<&Self>
    where
        P: Fn(&Self) -> bool,
    {
        if predicate(self) {
            vec![self]
        } else if let Some((children, _)) = &self.children {
            children.iter().flat_map(|child| child.select_clusters(predicate)).collect()
        } else {
            vec![]
        }
    }

    /// Returns mutable references to the clusters in the subtree rooted here that satisfy the given predicate.
    ///
    /// Once the predicate returns `true` for a cluster, its subtree is not searched further.
    pub fn select_clusters_mut<P>(&mut self, predicate: &P) -> Vec<&mut Self>
    where
        P: Fn(&Self) -> bool,
    {
        if predicate(self) {
            vec![self]
        } else if let Some((children, _)) = &mut self.children {
            children.iter_mut().flat_map(|child| child.select_clusters_mut(predicate)).collect()
        } else {
            vec![]
        }
    }
}
