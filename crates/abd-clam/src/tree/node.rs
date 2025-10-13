//! A `Node` in a `Tree` for use in CLAM.

/// A node in the `Tree`.
pub struct Node<T, A> {
    /// Depth of this node in the tree, with root at depth 0.
    pub(crate) depth: usize,
    /// Index of the center item in the `items` array of the `Tree`.
    pub(crate) center_index: usize,
    /// Number of items in the subtree rooted at this node, including the center item.
    pub(crate) cardinality: usize,
    /// The distance from the center item to the furthest item in the subtree.
    pub(crate) radius: T,
    /// The Local Fractal Dimension of the `Node`.
    pub(crate) lfd: f64,
    /// The children and span of this node, if it was partitioned. The span is the distance between the two poles used to partition the node.
    pub(crate) children: Option<(Box<[Self]>, T)>,
    /// Optional arbitrary data associated with this node.
    pub(crate) annotation: Option<A>,
}

impl<T, A> Node<T, A> {
    /// Returns the depth of this node in the tree.
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the index of the center item in the `items` array of the `Tree`.
    pub const fn center_index(&self) -> usize {
        self.center_index
    }

    /// Returns the number of items in the subtree rooted at this node, including the center item.
    pub const fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Returns a `Range` that can be used to index into the `items` array of the `Tree` for all items in the subtree rooted at this node, excluding the center
    /// item of this node.
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

    /// Returns the Local Fractal Dimension (LFD) of this node.
    pub const fn lfd(&self) -> f64 {
        self.lfd
    }

    /// Returns true if this node is a singleton (i.e., contains exactly one item or has a radius of zero).
    pub fn is_singleton(&self) -> bool
    where
        T: num_traits::Zero,
    {
        self.cardinality == 1 || self.radius.is_zero()
    }

    /// Returns true if this node is a leaf (i.e., has no children).
    pub const fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// Returns a reference to the children of this node, if any.
    ///
    /// Use this to read the subtree rooted at this node without taking ownership.
    pub fn children(&self) -> Option<&[Self]> {
        self.children.as_ref().map(|(children, _)| children.as_ref())
    }

    /// Returns the span of this node, if it has children.
    ///
    /// The span is the distance between the two poles used to partition the node.
    pub fn span(&self) -> Option<&T> {
        self.children.as_ref().map(|(_, span)| span)
    }

    /// Returns true if this node has an annotation.
    pub const fn is_annotated(&self) -> bool {
        self.annotation.is_some()
    }

    /// Returns an optional reference to the node's annotation, if any.
    ///
    /// Use this to read metadata attached to the node without taking ownership.
    pub const fn annotation(&self) -> Option<&A> {
        self.annotation.as_ref()
    }

    /// Returns an optional mutable reference to the node's annotation, if any.
    ///
    /// Use this to modify node metadata in place.
    pub const fn annotation_mut(&mut self) -> Option<&mut A> {
        self.annotation.as_mut()
    }

    /// Removes and returns the node's annotation, if any, leaving it unannotated.
    pub const fn take_annotation(&mut self) -> Option<A> {
        self.annotation.take()
    }

    /// Annotates the node with the given value, replacing any existing annotation.
    pub fn annotate(&mut self, annotation: A) {
        self.annotation = Some(annotation);
    }

    /// Annotates the node by evaluating the given closure, returning any existing annotation.
    ///
    /// The closure is called with a reference to the node and a reference to the previous annotation (if any).
    pub fn annotate_with<F: FnOnce(&Self, Option<&A>) -> A>(&mut self, f: F) -> Option<A> {
        let old_annotation = self.annotation.take();
        self.annotation = Some(f(self, old_annotation.as_ref()));
        old_annotation
    }

    /// Changes the annotations, and their types, of this node and all its descendants by applying the given closure recursively in pre-order.
    ///
    /// The closure is called with a reference to each node before its children are processed and its return value becomes the new annotation for that node.
    pub fn annotate_pre_order<B, F: FnMut(&Self) -> Option<B>>(mut self, f: &mut F) -> Node<T, B> {
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

        Node {
            depth: self.depth,
            center_index: self.center_index,
            cardinality: self.cardinality,
            radius: self.radius,
            lfd: self.lfd,
            children,
            annotation,
        }
    }

    /// Changes the annotations, and their types, of this node and all its descendants by applying the given closure recursively in post-order.
    ///
    /// The closure is called with a reference to each node after its children have been processed, along with the previous annotation (if any). The closure's
    /// return value becomes the new annotation for that node.
    pub fn annotate_post_order<B, F: FnMut(&Node<T, B>, Option<A>) -> Option<B>>(mut self, f: &mut F) -> Node<T, B> {
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

        let mut node = Node {
            depth: self.depth,
            center_index: self.center_index,
            cardinality: self.cardinality,
            radius: self.radius,
            lfd: self.lfd,
            children,
            annotation: None,
        };

        node.annotation = f(&node, old_annotation);

        node
    }
}
