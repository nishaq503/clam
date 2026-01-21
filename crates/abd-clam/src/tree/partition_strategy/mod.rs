//! How a `Cluster` is partitioned into child clusters.

use super::Cluster;

mod branching_factor;
mod min_split;
mod span_reduction_factor;

pub use branching_factor::BranchingFactor;
pub use min_split::MinSplit;
pub use span_reduction_factor::SpanReductionFactor;

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
    /// The minimum size of the smaller child cluster when partitioning a cluster.
    pub(crate) min_split: MinSplit,
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
            min_split: MinSplit::None,
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
            min_split: MinSplit::None,
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
            min_split: self.min_split,
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

    /// Sets the minimum size of the smaller child cluster when partitioning a cluster.
    ///
    /// If set to `MinSplit::Fixed` with a value outside the range `(0.0, 0.5]`, it defaults to `MinSplit::None`.
    pub fn with_min_split(mut self, min_split: MinSplit) -> Self {
        self.min_split = if let MinSplit::Fixed(fraction) = min_split {
            if (0.0 < fraction) && (fraction <= 0.5) {
                MinSplit::Fixed(fraction)
            } else {
                MinSplit::None
            }
        } else {
            min_split
        };
        self
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
