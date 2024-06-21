//! An `Edge` connects two `Cluster`es in a `Graph`. All `Edge`s are bidirectional.

use distances::Number;

use crate::Cluster;

/// An `Edge` connects two `Cluster`es in a `Graph`. All `Edge`s are bidirectional.
#[derive(Debug, Clone)]
pub struct Edge<'a, U: Number, C: Cluster<U>> {
    /// A reference to the first `Cluster` connected by this `Edge`.
    left: &'a C,
    /// A reference to the second `Cluster` connected by this `Edge`.
    right: &'a C,
    /// The distance between the two `Cluster`s' centers.
    distance: U,
}

impl<'a, U: Number, C: Cluster<U>> Edge<'a, U, C> {
    /// Creates a new `Edge` between the given `Cluster`s with the specified distance.
    ///
    /// The user should ensure that the provided `Cluster`s are appropriately related to have an edge
    /// between them. The `Edge` is always created as a bi-directional edge between the two `Cluster`s.
    ///
    /// # Arguments
    ///
    /// * `left`: The first `Cluster` connected by the `Edge`.
    /// * `right`: The second `Cluster` connected by the `Edge`.
    /// * `distance`: The distance between the two `Cluster`s.
    ///
    /// # Returns
    ///
    /// A new `Edge` connecting the provided `Cluster`s with the given distance.
    pub fn new(left: &'a C, right: &'a C, distance: U) -> Self {
        let (left, right) = if left < right { (left, right) } else { (right, left) };
        Self { left, right, distance }
    }

    /// Checks if this edge contains the given `Cluster` at one of its ends.
    ///
    /// # Arguments
    ///
    /// * `c`: The `Cluster` to check if it's at one of the ends of the edge.
    ///
    /// # Returns
    ///
    /// Returns `true` if the `Cluster` is found at either end of the edge, `false` otherwise.
    pub fn contains(&self, c: &C) -> bool {
        c == self.left || c == self.right
    }

    /// Returns a 2-slice containing the `Cluster`s at the two ends of this `Edge`.
    ///
    /// # Returns
    ///
    /// A 2-slice containing the `Cluster`s at the left and right ends of the edge.
    pub const fn clusters(&self) -> [&C; 2] {
        [self.left, self.right]
    }

    /// Retrieves a reference to the `Cluster` at the `left` end of the `Edge`.
    ///
    /// # Returns
    ///
    /// A reference to the `Cluster` at the left end of the `Edge`.
    pub const fn left(&self) -> &C {
        self.left
    }

    /// Retrieves a reference to the `Cluster` at the `right` end of the `Edge`.
    ///
    /// # Returns
    ///
    /// A reference to the `Cluster` at the right end of the `Edge`.
    pub const fn right(&self) -> &C {
        self.right
    }

    /// Gets the distance between the two `Cluster`s connected by this `Edge`.
    ///
    /// # Returns
    ///
    /// The distance value representing the length between the two connected clusters.
    pub const fn distance(&self) -> U {
        self.distance
    }

    /// Checks whether this is an edge from a `Cluster` to itself.
    ///
    /// # Returns
    ///
    /// - `true` if the edge connects a `Cluster` to itself, indicating a circular relationship.
    /// - `false` if the edge connects two distinct clusters.
    pub fn is_circular(&self) -> bool {
        self.left == self.right
    }

    /// Returns the neighbor of the given `Cluster` in this `Edge`.
    ///
    /// # Arguments
    ///
    /// * `c`: The `Cluster` for which to find the neighbor.
    ///
    /// # Returns
    ///
    /// A reference to the neighboring `Cluster` connected by this `Edge`.
    ///
    /// # Errors
    ///
    /// Returns an error if `c` is not one of the `Cluster`s connected by this `Edge`.
    pub fn neighbor(&self, c: &C) -> Result<&C, String> {
        if c == self.left {
            Ok(self.right)
        } else if c == self.right {
            Ok(self.left)
        } else {
            Err(format!("Cluster {c} is not in this edge {self:?}."))
        }
    }
}

impl<'a, U: Number, C: Cluster<U>> PartialEq for Edge<'a, U, C> {
    fn eq(&self, other: &Self) -> bool {
        (self.left == other.left) && (self.right == other.right)
    }
}

/// Two `Edge`s are equal if they connect the same two `Cluster`s.
impl<'a, U: Number, C: Cluster<U>> Eq for Edge<'a, U, C> {}

impl<'a, U: Number, C: Cluster<U>> std::fmt::Display for Edge<'a, U, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:} -- {:}", self.left, self.right)
    }
}

impl<'a, U: Number, C: Cluster<U>> core::hash::Hash for Edge<'a, U, C> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        format!("{self}").hash(state);
    }
}
