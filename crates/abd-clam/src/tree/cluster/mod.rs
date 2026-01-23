//! A `Cluster` in a `Tree` for use in CLAM.

mod getters;
mod setters;
#[cfg(feature = "serde")]
mod to_csv;
mod traversal;

/// A `Cluster` is a node in the `Tree` that represents a subset of the items in the `Tree`.
///
/// It contains information about the subset including:
///
/// - `depth`: The depth of the cluster in the tree, with the root at depth 0.
/// - `center_index`: The index of the center item in the `items` array of the `Tree`.
/// - `cardinality`: The number of items in the subtree rooted at this cluster, including the center item.
/// - `radius`: The distance from the center item to the furthest item in the `Cluster`.
/// - `lfd`: The Local Fractal Dimension of the `Cluster`.
/// - `children`: The children of this cluster, if it was partitioned.
/// - `span`: The distance between the two poles used to partition the cluster, if it was partitioned.
/// - `annotation`: Optional arbitrary data associated with this cluster.
///
/// # Generics
///
/// - `T`: The type of the distance values between items.
/// - `A`: The type of the annotation data associated with this cluster.
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
    /// Returns a cloned cluster without any annotations.
    pub fn clone_without_annotations<B>(&self) -> Cluster<T, B>
    where
        T: Clone,
    {
        // Post-order traversal stack
        let mut stack = {
            let mut stack_1 = vec![self];
            let mut stack_2 = Vec::new();

            while let Some(c) = stack_1.pop() {
                if let Some((children, _)) = &c.children {
                    stack_1.extend(children.iter());
                }
                stack_2.push(c);
            }
            stack_2
        };

        let mut cloned_children: Vec<Cluster<T, B>> = Vec::new();
        while let Some(c) = stack.pop() {
            let children = if let Some((children, span)) = &c.children {
                // The cloned children of `c` are the last `n_children` clusters in `cloned_children`.
                let n_children = children.len();
                let start_index = cloned_children.len() - n_children;
                let cloned_children_slice = cloned_children.split_off(start_index);
                Some((cloned_children_slice.into_boxed_slice(), span.clone()))
            } else {
                // Leaf cluster
                None
            };

            // Create the cloned cluster without annotation.
            let cloned_cluster = Cluster {
                depth: c.depth,
                center_index: c.center_index,
                cardinality: c.cardinality,
                radius: c.radius.clone(),
                lfd: c.lfd,
                children,
                annotation: None,
            };

            // Push the cloned cluster onto the stack.
            cloned_children.push(cloned_cluster);
        }

        cloned_children.pop().unwrap_or_else(|| unreachable!("The root cluster is always present"))
    }

    /// Clears the annotation of this cluster and all its descendants.
    pub fn clear_annotations<B>(self) -> Cluster<T, B> {
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
    pub fn filter_clusters<P, Args>(&self, predicate: &P, args: &Args) -> Vec<&Self>
    where
        P: Fn(&Self, &Args) -> bool,
    {
        if predicate(self, args) {
            vec![self]
        } else if let Some((children, _)) = &self.children {
            children.iter().flat_map(|child| child.filter_clusters(predicate, args)).collect()
        } else {
            vec![]
        }
    }

    /// Returns mutable references to the clusters in the subtree rooted here that satisfy the given predicate.
    ///
    /// Once the predicate returns `true` for a cluster, its subtree is not searched further.
    pub fn filter_clusters_mut<P, Args>(&mut self, predicate: &P, args: &Args) -> Vec<&mut Self>
    where
        P: Fn(&Self, &Args) -> bool,
    {
        if predicate(self, args) {
            vec![self]
        } else if let Some((children, _)) = &mut self.children {
            children.iter_mut().flat_map(|child| child.filter_clusters_mut(predicate, args)).collect()
        } else {
            vec![]
        }
    }
}
