//! A `Cluster` in a `Tree` for use in CLAM.

mod getters;
mod setters;
#[cfg(feature = "serde")]
mod to_csv;
// mod traversal;

// pub use traversal::AnnotatedItems;

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
    /// The indices of child centers and the span of this cluster, if it was partitioned. The span is the distance between the two poles used to partition the cluster.
    pub(crate) children: Option<(Box<[usize]>, T)>,
    /// Optional arbitrary data associated with this cluster.
    pub(crate) annotation: A,
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

        fields.push(format!("annotation: {:?}", self.annotation));

        let name = if let Some((child_center_indices, span)) = &self.children {
            fields.push(format!("span: {span}"));
            fields.push(format!("child_centers: {child_center_indices:?}"));

            "P"
        } else {
            "L"
        };

        write!(f, "{name}: {}", fields.join(", "))
    }
}
