//! How the number of children is determined when partitioning a cluster.

use crate::utils::MinItem;

/// The branching factor can be fixed, logarithmic, adaptive, or effectively unbounded (controlled by the [`SpanReductionFactor`]).
#[must_use]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub enum BranchingFactor {
    /// Each cluster can have zero or `n` children. If `n < 2`, it is treated as 2.
    Fixed(usize),
    /// A cluster with `n` non-center items will have `ceil(log n)` children.
    Logarithmic,
    /// We use some heuristics from our analysis of the expected ratio of the size of the subtree to the cardinality of the cluster, to select a branching
    /// factor that minimizes the expected size of the subtree. This branching factor is recomputed for each cluster based on the number of non-center items in
    /// that cluster.
    Adaptive(usize),
    /// The branching factor is effectively unbounded and will be controlled by the [`SpanReductionFactor`] (SRF).
    #[default]
    Unbounded,
}

impl std::fmt::Display for BranchingFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(k) => write!(f, "Fixed({k})"),
            Self::Logarithmic => write!(f, "Logarithmic"),
            Self::Adaptive(max_k) => write!(f, "Adaptive({max_k})"),
            Self::Unbounded => write!(f, "Unbounded"),
        }
    }
}

impl From<usize> for BranchingFactor {
    fn from(value: usize) -> Self {
        Self::Fixed(value.max(2))
    }
}

impl BranchingFactor {
    /// Returns the branching factor for a cluster with the given the cardinality of the cluster, if not `Unbounded`.
    #[expect(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    #[must_use]
    pub fn for_cardinality(&self, n: usize) -> Option<usize> {
        match self {
            Self::Fixed(b) => Some(*b),
            Self::Logarithmic if n >= 5 => {
                // Use a branching factor of O(log n), where n is the number of non-center items in the cluster.
                // Since `cardinality > 2`, this is at least 2.
                Some(((n - 1) as f64).log2().ceil() as usize)
            }
            Self::Logarithmic => Some(2), // For n < 5, just use a branching factor of 2
            Self::Adaptive(max_b) => (2..=*max_b)
                .map(|b| (b, expected_num_clusters(n, b) as f64 / n as f64))
                .min_by_key(|&(_, r)| MinItem((), r))
                .map(|(b, _)| b)
                .or(Some(2)),
            Self::Unbounded => None, // Effectively no limit on branching factor; SRF will control the actual branching factor
        }
    }
}

/// Recursively finds the number of clusters in a balanced tree with the given branching factor.
///
/// This implements the following recurrence relation:
///   - `T(1) = 1` and `T(2) = 1`, the leaf clusters.
///   - `T(n) = n - 1`, for `3 <= n <= b + 1`, the clusters whose children are all leaves
///   - `T(1 + a + b * n) = 1 + a * T(n + 1) + (b - a) * T(n)`, for `n > b + 1` and `0 <= a < b`.
///
/// This function is used to determine the number of children to create when the [`BranchingFactor`] is `Adaptive`. The chosen branching factor is the value of
/// `b` that minimizes the expected ratio of the size of the subtree to the cardinality of the cluster.
///
/// # Arguments
///
/// - `n`: The cardinality of the cluster (number of items, including the center).
/// - `b`: The branching factor of the tree.
#[must_use]
pub fn expected_num_clusters(n: usize, b: usize) -> usize {
    if n < 3 {
        1
    } else if n < b + 2 {
        n - 1
    } else {
        let a = (n - 1) % b;
        let n = (n - 1) / b;
        1 + a * expected_num_clusters(n + 1, b) + (b - a) * expected_num_clusters(n, b)
    }
}
