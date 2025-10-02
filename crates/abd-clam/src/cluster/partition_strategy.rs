//! Strategy for partitioning a `Cluster` into child clusters.

use num::Integer;

use crate::DistanceValue;

use super::Cluster;

/// Strategy for partitioning a `Cluster` into child clusters.
///
/// Our implementation of the hierarchical clustering algorithm may evolve over time. For now the strategy is used as follows:
///   - The `partition` method (and its variants) will first check the `predicate` to see if the cluster should be partitioned. If the `predicate` is satisfied,
///     we proceed to partition the cluster; otherwise, we leave the cluster as a leaf.
///   - If the `branching_factor` not `SRF`, we use the approach described in the [`BranchingFactor`] enum to determine how many children to create.
///   - If the `branching_factor` is `SRF`, we will use the approach described in the [`SpanReductionFactor`] enum to determine how many children to create.
///
/// The default `PartitionStrategy` will partition any cluster with more than one non-center item, using a span reduction factor of `√2`.
#[must_use]
pub struct PartitionStrategy<Id, I, T: DistanceValue, A, P: Fn(&Cluster<Id, I, T, A>) -> bool> {
    /// The predicate that determines whether a cluster should be partitioned into child clusters.
    pub(crate) predicate: P,
    /// The branching factor of the cluster tree.If the predicate is satisfied, the non-center items in the cluster are partitioned into subsets until the desired `branching_factor` is reached.
    ///   - If the `branching_factor` is `Adaptive`, then the `span_reduction` factor is used to further partition the subsets until all child clusters will likely
    ///     have a span that is small enough to satisfy the span reduction criterion.
    pub(crate) branching_factor: BranchingFactor,
    /// Span reduction factor.
    pub(crate) span_reduction: SpanReductionFactor,
    /// Ghost in the machine.
    phantom: core::marker::PhantomData<(Id, I, T, A)>,
}

impl<Id, I, T: DistanceValue, A, P: Fn(&Cluster<Id, I, T, A>) -> bool> std::fmt::Display
    for PartitionStrategy<Id, I, T, A, P>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if matches!(self.branching_factor, BranchingFactor::SRF) {
            write!(f, "SRF({})", self.span_reduction)
        } else {
            write!(f, "BF({})", self.branching_factor)
        }
    }
}

impl<Id, I, T: DistanceValue, A> Default
    for PartitionStrategy<Id, I, T, A, Box<dyn Fn(&Cluster<Id, I, T, A>) -> bool + Send + Sync>>
{
    fn default() -> Self {
        Self {
            predicate: Box::new(|b: &Cluster<Id, I, T, A>| b.cardinality > 2),
            branching_factor: BranchingFactor::default(),
            span_reduction: SpanReductionFactor::default(),
            phantom: core::marker::PhantomData,
        }
    }
}

impl<Id, I, T: DistanceValue, A, P: Fn(&Cluster<Id, I, T, A>) -> bool> PartitionStrategy<Id, I, T, A, P> {
    /// Creates a new `PartitionStrategy` with the given predicate.
    pub fn new(predicate: P) -> Self {
        Self {
            predicate,
            branching_factor: BranchingFactor::default(),
            span_reduction: SpanReductionFactor::default(),
            phantom: core::marker::PhantomData,
        }
    }

    /// Sets the predicate that determines whether a cluster should be partitioned into child clusters.
    ///
    /// The default predicate will allow partitioning of any cluster with more than one non-center item.
    pub fn with_predicate(mut self, predicate: P) -> Self {
        self.predicate = predicate;
        self
    }

    /// Sets the branching factor of the cluster tree.
    ///
    /// The default branching factor is `SRF`.
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

/// The branching factor can be fixed, logarithmic, adaptive, or effectively unbounded (controlled by the `SpanReductionFactor`).
#[derive(Debug)]
#[non_exhaustive]
pub enum BranchingFactor {
    /// Each cluster can have zero or `n` children. If `n < 2`, it is treated as 2.
    Fixed(usize),
    /// A cluster with `n` non-center items can have between 2 and `ceil(log n)` children.
    Logarithmic,
    /// The branching factor is adaptively chosen (with a minimum of 2 and a maximum of the given value) to minimize the expected ratio of the size of the
    /// subtree to the cardinality of the cluster.
    Adaptive(usize),
    /// The branching factor is effectively unbounded; the actual branching factor will be controlled by the `SpanReductionFactor` (SRF).
    SRF,
}

impl std::fmt::Display for BranchingFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(k) => write!(f, "Fixed({k})"),
            Self::Logarithmic => write!(f, "Logarithmic"),
            Self::Adaptive(max_k) => write!(f, "Adaptive({max_k})"),
            Self::SRF => write!(f, "SRF"),
        }
    }
}

impl BranchingFactor {
    /// Returns the branching factor for a cluster with the given the cardinality of the cluster, if not `SRF`.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    #[must_use]
    pub fn for_cardinality(&self, cardinality: usize) -> Option<usize> {
        match self {
            Self::Fixed(k) => Some(*k),
            Self::Logarithmic => {
                // Use a branching factor of O(log n), where n is the number of non-center items in the cluster.
                // Since `cardinality > 2`, this is at least 2.
                Some(((cardinality - 1) as f64).log2().ceil() as usize)
            }
            Self::Adaptive(max_k) => (2..=*max_k)
                .map(|k| (k, expected_num_clusters(cardinality, k) as f64 / cardinality as f64))
                .filter(|&(_, r)| r < 0.6)
                .min_by_key(|&(k, _)| k)
                .map(|(k, _)| k)
                .or(Some(2)),
            Self::SRF => None, // Effectively no limit on branching factor; SRF will control the actual branching factor
        }
    }
}

impl Default for BranchingFactor {
    fn default() -> Self {
        Self::SRF
    }
}

impl From<usize> for BranchingFactor {
    fn from(value: usize) -> Self {
        Self::Fixed(value.max(2))
    }
}

/// The Span Reduction Factor (SRF) of a `Cluster` controls how much the span of child clusters should be reduced compared to their parent cluster.
///
/// The `span` of a cluster is the distance between any two of its extremal items, e.g. the distance between poles used for partitioning in a binary tree. This
/// can be thought of as an analog to the diameter of a covering sphere in arbitrary metric (or non-metric) spaces. The SRF is the factor by which the span of
/// child clusters should be reduced compared to their parent. For example, `SpanReductionFactor::Two` means that the span of each child cluster should be at
/// most half the span of its parent.
#[non_exhaustive]
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
    pub fn max_child_span_for<T: DistanceValue>(&self, span: T) -> T {
        let factor = match self {
            Self::Fixed(srf) => srf.recip(),
            Self::Sqrt2 => core::f64::consts::FRAC_1_SQRT_2,
            Self::Two => 0.5,
            Self::E => core::f64::consts::E.recip(),
            Self::Pi => core::f64::consts::FRAC_1_PI,
            Self::Phi => crate::utils::PHI_F64.recip(),
        };
        let span = span
            .to_f64()
            .unwrap_or_else(|| unreachable!("DistanceValue must be convertible to f64"));
        T::from_f64(span * factor).unwrap_or_else(|| unreachable!("DistanceValue must be convertible from f64"))
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

/// Recursively finds the number of clusters in a balanced tree with the given branching factor.
///
/// This implements the following recurrence relation:
///   - `R(1) = 1` and `R(2) = 1`, the leaf clusters.
///   - `R(n) = n - 1`, for `3 <= n <= k + 1`, the parents of leaf clusters.
///   - `R(1 + i + k * n) = 1 + i * R(n + 1) + (k - i) * R(n)`, for `n > k + 1` and `0 <= i < k`.
///
/// This function is used to determine the number of children to create when the [`BranchingFactor`] is `Adaptive`. The chosen branching factor is the value of
/// `k` that minimizes the expected ratio of the size of the subtree to the cardinality of the cluster.
///
/// # Arguments
///
/// - `n`: The cardinality of the cluster (number of items, including the center).
/// - `k`: The branching factor of the tree.
#[must_use]
pub fn expected_num_clusters(n: usize, k: usize) -> usize {
    if n < 3 {
        1
    } else if n < k + 2 {
        n - 1
    } else {
        let (i, n) = (n - 1).div_rem(&k);
        1 + i * expected_num_clusters(n + 1, k) + (k - i) * expected_num_clusters(n, k)
    }
}
