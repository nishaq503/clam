//! How a `Cluster` is partitioned into child clusters.

#![allow(clippy::derivable_impls)]

use crate::{DistanceValue, utils::MinItem};

use super::Cluster;

/// How a `Cluster` is partitioned into child clusters.
///
/// The default `PartitionStrategy` will partition any cluster with more than one non-center item to make a binary tree whose leaves are all clusters with too
/// few items to be partitioned any further.
///
/// The strategy is controlled by the `predicate`, `branching_factor`, and `span_reduction` fields. It proceeds as follows:
///   - Given some items, we first create a Cluster as a leaf.
///   - We then evaluate the `predicate` to determine whether the cluster should be partitioned. If the `predicate` is satisfied, we proceed to partition the
///     cluster; otherwise, we leave the cluster as a leaf.
///   - Next, if the `branching_factor` is not `Unbounded`, we use the the [`BranchingFactor`] enum to determine how many children to create.
///   - Otherwise, if the `branching_factor` is `Unbounded`, we use the [`SpanReductionFactor`] enum to determine how many children to create.
///
/// Note that our implementation of `PartitionStrategy` may change in the future to improve performance or to add new features.
///
/// # Type Parameters
///
/// - `P`: A function: `&Cluster<T, A> -> bool` that determines whether a cluster should be partitioned into child clusters.
#[must_use]
#[derive(Debug, Clone, Copy)]
pub struct PartitionStrategy<P> {
    /// The predicate that determines whether a cluster should be partitioned into child clusters.
    pub(crate) predicate: P,
    /// The branching factor of the cluster tree.
    pub(crate) branching_factor: BranchingFactor,
    /// Span reduction factor.
    pub(crate) span_reduction: SpanReductionFactor,
}

impl<P> std::fmt::Display for PartitionStrategy<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if matches!(self.branching_factor, BranchingFactor::Unbounded) {
            write!(f, "SRF({})", self.span_reduction)
        } else {
            write!(f, "BF({})", self.branching_factor)
        }
    }
}

impl<T, A> Default for PartitionStrategy<fn(&Cluster<T, A>) -> bool> {
    fn default() -> Self {
        Self {
            predicate: |b: &Cluster<T, A>| b.cardinality > 2,
            branching_factor: BranchingFactor::Fixed(2),
            span_reduction: SpanReductionFactor::default(),
        }
    }
}

impl<P> PartitionStrategy<P> {
    /// Creates a new `PartitionStrategy` with the given predicate.
    pub fn new(predicate: P) -> Self {
        Self {
            predicate,
            branching_factor: BranchingFactor::default(),
            span_reduction: SpanReductionFactor::default(),
        }
    }

    /// Sets the predicate that determines whether a cluster should be partitioned into child clusters.
    ///
    /// The default predicate will allow partitioning of any cluster with more than one non-center item.
    pub const fn with_predicate<Q>(&self, predicate: Q) -> PartitionStrategy<Q> {
        PartitionStrategy {
            predicate,
            branching_factor: self.branching_factor,
            span_reduction: self.span_reduction,
        }
    }

    /// Evaluates the predicate on the given cluster.
    pub fn should_partition<T, A>(&self, cluster: &Cluster<T, A>) -> bool
    where
        P: Fn(&Cluster<T, A>) -> bool,
    {
        (self.predicate)(cluster)
    }

    /// Evaluates the predicate on the given cluster in parallel.
    pub fn par_should_partition<T, A>(&self, cluster: &Cluster<T, A>) -> bool
    where
        T: Send + Sync,
        A: Send + Sync,
        P: Fn(&Cluster<T, A>) -> bool + Send + Sync,
    {
        (self.predicate)(cluster)
    }

    /// Sets the branching factor of the cluster tree.
    ///
    /// The default branching factor is `Unbounded`.
    pub fn with_branching_factor(mut self, branching_factor: BranchingFactor) -> Self {
        self.branching_factor = if let BranchingFactor::Fixed(n) = branching_factor {
            BranchingFactor::Fixed(n.max(2))
        } else if let BranchingFactor::Adaptive(max_k) = branching_factor {
            BranchingFactor::Adaptive(max_k.max(3))
        } else {
            branching_factor
        };
        self
    }

    /// Sets the span reduction factor (SRF).
    ///
    /// The default SRF is `Sqrt2`.
    pub fn with_span_reduction(mut self, span_reduction: SpanReductionFactor) -> Self {
        self.span_reduction = if let SpanReductionFactor::Fixed(srf) = span_reduction {
            srf.into()
        } else {
            span_reduction
        };
        self
    }
}

impl<'a, T, A> From<&'a str> for PartitionStrategy<fn(&Cluster<T, A>) -> bool> {
    fn from(value: &'a str) -> Self {
        let strategy = Self::default();
        match value.to_lowercase().as_str() {
            srf if srf.starts_with("srf(") && srf.ends_with(')') => {
                let inner = &srf["srf(".len()..srf.len() - 1];
                strategy.with_span_reduction(SpanReductionFactor::from(inner))
            }
            bf if bf.starts_with("bf(") && bf.ends_with(')') => {
                let inner = &bf["bf(".len()..bf.len() - 1];
                strategy.with_branching_factor(BranchingFactor::from(inner))
            }
            _ => strategy,
        }
    }
}

impl<T, A> From<String> for PartitionStrategy<fn(&Cluster<T, A>) -> bool> {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

/// The branching factor can be fixed, logarithmic, adaptive, or effectively unbounded (controlled by the [`SpanReductionFactor`]).
#[must_use]
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
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

impl Default for BranchingFactor {
    fn default() -> Self {
        Self::Unbounded
    }
}

impl From<usize> for BranchingFactor {
    fn from(value: usize) -> Self {
        Self::Fixed(value.max(2))
    }
}

impl<'a> From<&'a str> for BranchingFactor {
    fn from(value: &'a str) -> Self {
        match value.to_lowercase().as_str() {
            "logarithmic" => Self::Logarithmic,
            "unbounded" => Self::Unbounded,
            adaptive if adaptive.starts_with("adaptive(") && adaptive.ends_with(')') => {
                let inner = &adaptive["adaptive(".len()..adaptive.len() - 1];
                inner.parse::<usize>().map_or(Self::Adaptive(128), |n| Self::Adaptive(n.max(3)))
            }
            fixed if fixed.starts_with("fixed(") && fixed.ends_with(')') => {
                let inner = &fixed["fixed(".len()..fixed.len() - 1];
                inner.parse::<usize>().map_or(Self::Fixed(2), |n| Self::Fixed(n.max(2)))
            }
            _ => value.parse::<usize>().map_or(Self::Fixed(2), |n| Self::Fixed(n.max(2))),
        }
    }
}

impl From<String> for BranchingFactor {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

/// The Span Reduction Factor (SRF) of a `Cluster` controls how much the span of child clusters should be reduced compared to their parent cluster.
///
/// The `span` of a cluster is the distance between any two of its extremal items, e.g. the distance between poles used for partitioning in a binary tree. This
/// can be thought of as an analog to the diameter of a covering sphere in arbitrary metric (or non-metric) spaces. The SRF is the factor by which the span of
/// child clusters should be reduced compared to their parent. For example, `SpanReductionFactor::Two` means that the span of each child cluster should be at
/// most half the span of its parent.
#[must_use]
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum SpanReductionFactor {
    /// Use a fixed SRF value. This must be in the range (1, ∞). If the value is outside this range, the SRF defaults to `√2`.
    Fixed(f64),
    /// The SRF is `√2`.
    Sqrt2,
    /// The SRF is `2`.
    Two,
    /// The SRF is `e`.
    E,
    /// The SRF is `π`.
    Pi,
    /// The SRF is the golden ratio `φ = (1 + √5) / 2`.
    Phi,
}

impl SpanReductionFactor {
    /// Returns the maximum allowed child span for a given span from the parent cluster.
    pub fn max_child_span_for<T: DistanceValue>(&self, parent_span: T) -> T {
        let factor = match self {
            Self::Fixed(srf) => *srf,
            Self::Sqrt2 => core::f64::consts::SQRT_2,
            Self::Two => 2.0,
            Self::E => core::f64::consts::E,
            Self::Pi => core::f64::consts::PI,
            Self::Phi => crate::utils::PHI_F64,
        };
        let parent_span = parent_span.to_f64().unwrap_or_else(|| unreachable!("DistanceValue must be convertible to f64"));
        T::from_f64(parent_span / factor).unwrap_or_else(|| unreachable!("DistanceValue must be convertible from f64"))
    }
}

impl Default for SpanReductionFactor {
    fn default() -> Self {
        Self::Sqrt2
    }
}

impl std::fmt::Display for SpanReductionFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(srf) => write!(f, "Fixed({srf})"),
            Self::Sqrt2 => write!(f, "Sqrt2"),
            Self::Two => write!(f, "Two"),
            Self::E => write!(f, "E"),
            Self::Pi => write!(f, "Pi"),
            Self::Phi => write!(f, "Phi"),
        }
    }
}

impl From<f64> for SpanReductionFactor {
    fn from(value: f64) -> Self {
        // We allow more tolerance when setting the SRF to common constants.
        if (value - core::f64::consts::SQRT_2).abs() < f64::EPSILON.sqrt() {
            Self::Sqrt2
        } else if (value - 2.0).abs() < f64::EPSILON.sqrt() {
            Self::Two
        } else if (value - core::f64::consts::E).abs() < f64::EPSILON.sqrt() {
            Self::E
        } else if (value - core::f64::consts::PI).abs() < f64::EPSILON.sqrt() {
            Self::Pi
        } else if (value - crate::utils::PHI_F64).abs() < f64::EPSILON.sqrt() {
            Self::Phi
        } else if 1.0 < value && value.is_finite() {
            Self::Fixed(value)
        } else {
            Self::Sqrt2 // Default to Sqrt2 if out of range
        }
    }
}

impl From<f32> for SpanReductionFactor {
    fn from(value: f32) -> Self {
        // We allow more tolerance when setting the SRF to common constants.
        if (value - core::f32::consts::SQRT_2).abs() < f32::EPSILON.sqrt() {
            Self::Sqrt2
        } else if (value - 2.0).abs() < f32::EPSILON.sqrt() {
            Self::Two
        } else if (value - core::f32::consts::E).abs() < f32::EPSILON.sqrt() {
            Self::E
        } else if (value - core::f32::consts::PI).abs() < f32::EPSILON.sqrt() {
            Self::Pi
        } else if (value - crate::utils::PHI_F32).abs() < f32::EPSILON.sqrt() {
            Self::Phi
        } else if 1.0 < value && value.is_finite() {
            Self::Fixed(f64::from(value))
        } else {
            Self::Sqrt2 // Default to Sqrt2 if out of range
        }
    }
}

impl<'a> From<&'a str> for SpanReductionFactor {
    fn from(value: &'a str) -> Self {
        match value.to_lowercase().as_str() {
            "sqrt2" => Self::Sqrt2,
            "two" => Self::Two,
            "e" => Self::E,
            "pi" => Self::Pi,
            "phi" => Self::Phi,
            fixed if fixed.starts_with("fixed(") && fixed.ends_with(')') => {
                let inner = &fixed["fixed(".len()..fixed.len() - 1];
                inner
                    .parse::<f64>()
                    .map_or(Self::Sqrt2, |srf| if 1.0 < srf && srf.is_finite() { Self::Fixed(srf) } else { Self::Sqrt2 })
            }
            _ => value.parse::<f64>().map_or(Self::Sqrt2, Self::from),
        }
    }
}

impl From<String> for SpanReductionFactor {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
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
