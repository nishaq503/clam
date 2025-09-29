//! A `Cluster` can be partitioned into a tree.

use rayon::prelude::*;

use crate::{
    utils::{MaxItem, SizedHeap},
    DistanceValue,
};

use super::{Cluster, Contents};

/// Strategy for partitioning a `Cluster` into child clusters.
///
/// Our implementation of the heiarchical may evolve over time. For now the this strategy is used as follows:
///   - The `partition` method (and its variants) will first check the `predicate` to see if the cluster should be partitioned. If the `predicate` is satisfied,
///     we proceed to partition the cluster; otherwise, we leave the cluster as a leaf.
///   - If the `branching_factor` not `Adaptive`, we will partition the cluster into at most `branching_factor` children.
///   - If the `branching_factor` is `Adaptive`, we will use the `span_reduction` factor to determine how many children to create. We will continue partitioning
///     the cluster until the span of each child would likely be small enough to satisfy the `span_reduction` factor.
#[must_use]
#[non_exhaustive]
pub struct PartitionStrategy<Id, I, T: DistanceValue, A, P: Fn(&Cluster<Id, I, T, A>) -> bool> {
    /// The predicate that determines whether a cluster should be partitioned into child clusters.
    predicate: P,
    /// The branching factor of the cluster tree.If the predicate is satisfied, the non-center items in the cluster are partitioned into subsets until the desired `branching_factor` is reached.
    ///   - If the `branching_factor` is `Adaptive`, then the `span_reduction` factor is used to further partition the subsets until all child clusters will likely
    ///     have a span that is small enough to satisfy the span reduction criterion.
    branching_factor: BranchingFactor,
    /// Span reduction factor.
    span_reduction: SpanReductionFactor,
    /// Ghost in the machine.
    phantom: core::marker::PhantomData<(Id, I, T, A)>,
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
    /// The default predicate will allow partitioning of any cluster with more than two or more non-center items.
    pub fn with_predicate(mut self, predicate: P) -> Self {
        self.predicate = predicate;
        self
    }

    /// Sets the branching factor of the cluster tree.
    ///
    /// The default branching factor is `Adaptive`.
    pub fn with_branching_factor(mut self, branching_factor: BranchingFactor) -> Self {
        self.branching_factor = if let BranchingFactor::Fixed(n) = branching_factor {
            BranchingFactor::Fixed(n.max(2))
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

/// The branching factor of a `Cluster` controls how many child clusters a parent cluster can have.
#[non_exhaustive]
pub enum BranchingFactor {
    /// Each cluster can have up to `n` children. If `n < 2`, it is treated as 2.
    Fixed(usize),
    /// A cluster with `n` non-center items can have up to `O(log n)` children.
    Logarithmic,
    /// The branching factor is chosen adaptively based on the SRF used for partitioning.
    Adaptive,
}

impl Default for BranchingFactor {
    fn default() -> Self {
        Self::Adaptive
    }
}

impl From<usize> for BranchingFactor {
    fn from(value: usize) -> Self {
        Self::Fixed(value.max(2))
    }
}

impl BranchingFactor {
    /// Returns the branching factor for a cluster with the given the cardinality of the cluster.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    fn for_cardinality(&self, cardinality: usize) -> Option<usize> {
        match self {
            Self::Fixed(n) => Some(*n),
            Self::Logarithmic => {
                // Use a branching factor of O(log n), where n is the number of non-center items in the cluster.
                // Since `cardinality > 2`, this is at least 2.
                Some(((cardinality - 1) as f64).log2().ceil() as usize)
            }
            Self::Adaptive => None, // Effectively no limit on branching factor; SRF will control the actual branching factor
        }
    }
}

/// The Span Reduction Factor (SRF) of a `Cluster` controls how much the span of child clusters should be reduced compared to their parent cluster.
///
/// The `span` of a cluster is the distance between any two of its extremal items, e.g. the distance between poles used for partitioning in a binary tree. This
/// can be thought of as an analog to the diameter of a cluster in arbitrary metric (or non-metric) spaces. The SRF is the factor by which the span of child
/// clusters should be reduced compared to their parent. For example, `SpanReductionFactor::Two` means that the span of each child cluster should be at most
/// half the span of its parent.
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

impl Default for SpanReductionFactor {
    fn default() -> Self {
        Self::Sqrt2
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

impl SpanReductionFactor {
    /// Returns the maximum allowed child span for a given span from the parent cluster.
    fn max_child_span_for<T: DistanceValue>(&self, span: T) -> T {
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

impl<Id, I, T: DistanceValue, A> Cluster<Id, I, T, A> {
    /// Creates a new tree of `Cluster`s.
    ///
    /// # Parameters
    ///
    /// * `items`: The items to be clustered into a tree of clusters.
    /// * `metric`: A function that computes the distance between two items.
    /// * `strategy`: The `PartitionStrategy` that controls how the tree is constructed.
    ///
    /// # Errors
    ///
    /// - If `items` is empty.
    pub fn new_tree<M: Fn(&I, &I) -> T, P: Fn(&Self) -> bool>(
        items: Vec<(Id, I)>,
        metric: &M,
        strategy: &PartitionStrategy<Id, I, T, A, P>,
    ) -> Result<Self, String> {
        if items.is_empty() {
            return Err("Cannot create a tree with no items".to_string());
        }
        Ok(Self::with_center_only(items, metric).partition(metric, strategy))
    }

    /// Private constructor for `Cluster`.
    ///
    /// WARNING: This function does only sets the `center` and `cardinality` fields correctly. Other fields are assigned placeholders and must be computed in
    /// `partition`.
    pub(crate) fn with_center_only<M: Fn(&I, &I) -> T>(mut items: Vec<(Id, I)>, metric: &M) -> Self {
        if items.len() == 1 {
            // A singleton cluster: `center` is the only item, `radius` is 0, `LFD` is 1
            let center = items.pop().unwrap_or_else(|| unreachable!("Cardinality is 1"));
            Self {
                cardinality: 1,
                center,
                radius: T::zero(),
                lfd: 1.0, // LFD of a singleton is _defined_ as 1
                radial_sum: T::zero(),
                span: T::zero(),
                contents: Contents::Leaf(Vec::new()),
                annotation: None,
            }
        } else {
            // Find and remove the `center`.
            let center = {
                // Use a subset of items to compute the geometric median for efficiency.
                let gm_sample = if items.len() < 100 {
                    &items[..]
                } else {
                    let num_samples = num_samples(items.len(), 100, 10_000);
                    &items[..num_samples]
                };
                // Remove and return the `center`.
                let gm_index = crate::utils::geometric_median(gm_sample, metric);
                items.swap_remove(gm_index)
            };

            Self {
                cardinality: items.len() + 1, // +1 for the `center`
                center,
                radius: T::max_value(), // Placeholder; to be computed in `partition`
                lfd: f64::MAX,          // Placeholder; to be computed in `partition`
                radial_sum: T::zero(),  // Placeholder; to be computed in `partition`
                span: T::max_value(),   // Placeholder; to be computed in `partition`
                contents: Contents::Leaf(items),
                annotation: None,
            }
        }
    }

    /// Partitions the cluster into two child clusters based on the provided `metric` and `strategy`, then recursively partitions the children.
    ///
    /// This method can be called multiple times on the same cluster with different strategys to refine or coarsen the partitioning.
    ///
    /// Calling this method with different `BranchingFactor`s will only change the branching factor of clusters that get re-partitioned; clusters that do not
    /// satisfy the predicate in the strategy will remain unchanged.
    ///
    /// # Parameters
    ///
    /// * `metric`: A function that computes the distance between two items.
    /// * `strategy`: The `PartitionStrategy` that controls how the tree is constructed.
    pub fn partition<M: Fn(&I, &I) -> T, P: Fn(&Self) -> bool>(
        mut self,
        metric: &M,
        strategy: &PartitionStrategy<Id, I, T, A, P>,
    ) -> Self {
        match self.contents {
            Contents::Leaf(items) => {
                if items.is_empty() {
                    // A singleton cluster: nothing to partition.
                    self.radius = T::zero();
                    self.lfd = 1.0; // LFD of a singleton is _defined_ as 1
                    self.contents = Contents::Leaf(Vec::new());
                    return self;
                }
                if items.len() == 1 {
                    // A cluster with one center and one item: nothing to partition.
                    self.radius = metric(&self.center.1, &items[0].1);
                    self.lfd = 1.0; // LFD of clusters with 2 items is _defined_ as 1
                    self.radial_sum = self.radius;
                    self.contents = Contents::Leaf(items);
                    return self;
                }

                // At this point, we have at least 2 items so radius computation is meaningful, and we can always remove two poles to partition with.
                // We have to first compute the radius and LFD because `with_center_only` does not compute them.

                // Compute the radius and LFD of the cluster.
                let radial_distances = items
                    .iter()
                    .map(|item| metric(&self.center.1, &item.1))
                    .collect::<Vec<_>>();
                let (arg_radius, radial_sum) = radial_distances
                    .iter()
                    .enumerate()
                    .fold((0, T::zero()), |(max_i, sum), (i, &d)| {
                        (if d > radial_distances[max_i] { i } else { max_i }, sum + d)
                    });
                self.radius = radial_distances[arg_radius];
                self.lfd = crate::utils::lfd_estimate(&radial_distances, self.radius);
                self.radial_sum = radial_sum;

                // Replace contents so that we can check the partition criteria without running into borrow issues.
                self.contents = Contents::Leaf(items);

                if !(strategy.predicate)(&self) {
                    // Criteria not satisfied; do not partition.
                    return self;
                }

                // Criteria are satisfied, so we partition the cluster.

                // Take ownership of the items for partitioning.
                let items = match core::mem::replace(&mut self.contents, Contents::Leaf(Vec::new())) {
                    Contents::Leaf(items) => items,
                    Contents::Children(_) => unreachable!("We just replaced contents with a leaf"),
                };

                // Split items into two groups based on their distances to two poles.
                let ([left, right], span) = split_by_poles(items, metric, Some(arg_radius));
                self.span = span;

                let mut splits = vec![left, right];
                if let Some(branching_factor) = strategy.branching_factor.for_cardinality(self.cardinality) {
                    // Further split the largest partition until we reach the desired branching factor.
                    while splits.len() < branching_factor {
                        let (largest_partition_index, _) = splits
                            .iter()
                            .enumerate()
                            .max_by_key(|&(_, (items, _))| items.len())
                            .unwrap_or_else(|| unreachable!("splits is non-empty"));
                        let (items, span) = splits.swap_remove(largest_partition_index);
                        if items.len() <= 2 {
                            // Cannot split further.
                            splits.push((items, span));
                            break;
                        }
                        let (split, _) = split_by_poles(items, metric, None);
                        splits.extend(split);
                    }
                } else {
                    // Use the SRF to continue splitting until the span reduction criterion is met.
                    let max_child_span = strategy.span_reduction.max_child_span_for(self.span);
                    let mut splits_heap = SizedHeap::from_iter(splits);
                    while splits_heap.peek().map_or_else(
                        || unreachable!("splits is non-empty"),
                        |(_, &span)| span > max_child_span,
                    ) {
                        // Further split the largest partition until all partitions meet the span reduction criterion.
                        let (items, span) = splits_heap.pop().unwrap_or_else(|| unreachable!("splits is non-empty"));
                        if items.len() <= 2 {
                            // Cannot split further.
                            splits_heap.push((items, span));
                            break;
                        }
                        let (split, _) = split_by_poles(items, metric, None);
                        splits_heap.extend(split);
                    }
                    splits = splits_heap.take_items().collect();
                }

                // Recursively create children and set the contents to the new children.
                self.contents = Contents::Children(
                    splits
                        .into_iter()
                        .map(|(child_items, _)| Self::with_center_only(child_items, metric))
                        .map(|child| child.partition(metric, strategy))
                        .map(Box::new)
                        .collect(),
                );
            }
            Contents::Children(children) => {
                // Replace contents so that we can check the partition criteria without running into borrow issues.
                self.contents = Contents::Children(children);

                if !(strategy.predicate)(&self) {
                    // Criteria not satisfied; convert back to a leaf by collecting all items from subtree.
                    self.contents = Contents::Leaf(self.take_subtree_items());
                }

                // Criteria are satisfied, so we continue checking children.
                // This is necessary because the user may have provided different criteria when the tree was last partitioned.

                // Take ownership of the children for partitioning.
                let children = match core::mem::replace(&mut self.contents, Contents::Leaf(Vec::new())) {
                    Contents::Children(children) => children,
                    Contents::Leaf(_) => unreachable!("We just replaced contents with children"),
                };

                // Recursively partition children and set the contents to the new children.
                self.contents = Contents::Children(
                    children
                        .into_iter()
                        .map(|child| child.partition(metric, strategy))
                        .map(Box::new)
                        .collect(),
                );
            }
        }

        // Return the (possibly) partitioned cluster.
        self
    }

    /// Removes and returns all items from the cluster and its descendants, excluding the center of this cluster; the children are dropped in the process and
    /// this cluster becomes a leaf with no items other than its center.
    ///
    /// WARNING: This function does not recompute any properties for three the up to the root after removing the items. The caller must ensure that the
    /// properties are still valid after this operation.
    pub(crate) fn take_subtree_items(&mut self) -> Vec<(Id, I)> {
        // Take ownership of the contents so we can recurse and drop children.
        let contents = core::mem::replace(&mut self.contents, Contents::Leaf(Vec::new()));
        match contents {
            Contents::Leaf(items) => items,
            Contents::Children(children) => {
                let mut items = Vec::with_capacity(self.cardinality - 1);
                for mut child in children {
                    items.extend(child.take_subtree_items());
                    items.push(child.center);
                }

                items
            }
        }
    }
}

impl<Id: Send + Sync, I: Send + Sync, T: DistanceValue + Send + Sync, A: Send + Sync> Cluster<Id, I, T, A> {
    /// Parallel version of [`new_tree`](Self::new_tree).
    ///
    /// # Errors
    ///
    /// - See [`new_tree`](Self::new_tree) for details.
    pub fn par_new_tree<M: Fn(&I, &I) -> T + Send + Sync, P: Fn(&Self) -> bool + Send + Sync>(
        items: Vec<(Id, I)>,
        metric: &M,
        strategy: &PartitionStrategy<Id, I, T, A, P>,
    ) -> Result<Self, String> {
        if items.is_empty() {
            return Err("Cannot create a Cluster tree with no items".to_string());
        }
        Ok(Self::par_with_center_only(items, metric).par_partition(metric, strategy))
    }

    /// Parallel version of [`with_center_only`](Self::with_center_only).
    pub(crate) fn par_with_center_only<M: Fn(&I, &I) -> T + Send + Sync>(mut items: Vec<(Id, I)>, metric: &M) -> Self {
        if items.len() == 1 {
            // A singleton cluster: center is the only item, radius is 0, LFD is 1
            let center = items.pop().unwrap_or_else(|| unreachable!("Cardinality is 1"));
            Self {
                cardinality: 1,
                center,
                radius: T::zero(),
                lfd: 1.0, // LFD of a singleton is _defined_ as 1
                radial_sum: T::zero(),
                span: T::zero(),
                contents: Contents::Leaf(Vec::new()),
                annotation: None,
            }
        } else {
            // Find and remove the center item.
            let center = {
                // Use a subset of items to compute the geometric median for efficiency.
                let gm_sample = if items.len() < 100 {
                    &items[..]
                } else {
                    let num_samples = num_samples(items.len(), 100, 10_000);
                    &items[..num_samples]
                };
                let gm_index = crate::utils::par_geometric_median(gm_sample, metric);
                // Remove and return the center item.
                items.swap_remove(gm_index)
            };

            Self {
                cardinality: items.len() + 1, // +1 for the center
                center,
                radius: T::max_value(), // Placeholder; to be computed in partition
                lfd: f64::MAX,          // Placeholder; to be computed in partition
                radial_sum: T::zero(),  // Placeholder; to be computed in partition
                span: T::max_value(),   // Placeholder; to be computed in partition
                contents: Contents::Leaf(items),
                annotation: None,
            }
        }
    }

    /// Parallel version of [`partition`](Self::partition).
    pub fn par_partition<M: Fn(&I, &I) -> T + Send + Sync, P: Fn(&Self) -> bool + Send + Sync>(
        mut self,
        metric: &M,
        strategy: &PartitionStrategy<Id, I, T, A, P>,
    ) -> Self {
        match self.contents {
            Contents::Leaf(items) => {
                if items.is_empty() {
                    // A singleton cluster: nothing to partition.
                    self.radius = T::zero();
                    self.lfd = 1.0; // LFD of a singleton is _defined_ as 1
                    self.contents = Contents::Leaf(Vec::new());
                    return self;
                }
                if items.len() == 1 {
                    // A cluster with one center and one item: nothing to partition.
                    self.radius = metric(&self.center.1, &items[0].1);
                    self.lfd = 1.0; // LFD of clusters with 2 items is _defined_ as 1
                    self.radial_sum = self.radius;
                    self.contents = Contents::Leaf(items);
                    return self;
                }

                // At this point, we have at least 2 items so radius computation is meaningful, and we can always remove two poles to partition with.
                // We have to first compute the radius and LFD because `new` does not compute them.

                // Compute the radius and LFD of the cluster.
                let radial_distances = items
                    .par_iter()
                    .map(|item| metric(&self.center.1, &item.1))
                    .collect::<Vec<_>>();
                let (arg_radius, radial_sum) = radial_distances
                    .iter()
                    .enumerate()
                    .fold((0, T::zero()), |(max_i, sum), (i, &d)| {
                        (if d > radial_distances[max_i] { i } else { max_i }, sum + d)
                    });
                self.radius = radial_distances[arg_radius];
                self.lfd = crate::utils::lfd_estimate(&radial_distances, self.radius);
                self.radial_sum = radial_sum;

                // Replace contents so that we can check the partition criteria without running into borrow issues.
                self.contents = Contents::Leaf(items);

                if !(strategy.predicate)(&self) {
                    // Criteria not satisfied; do not partition.
                    return self;
                }

                // Criteria are satisfied, so we partition the cluster.

                // Take ownership of the items for partitioning.
                let items = match core::mem::replace(&mut self.contents, Contents::Leaf(Vec::new())) {
                    Contents::Leaf(items) => items,
                    Contents::Children(_) => unreachable!("We just replaced contents with a leaf"),
                };

                // Split items into two groups based on their distances to two poles.
                let ([left, right], span) = par_split_by_poles(items, metric, Some(arg_radius));
                self.span = span;

                let mut splits = vec![left, right];
                if let Some(branching_factor) = strategy.branching_factor.for_cardinality(self.cardinality) {
                    // Further split the largest partition until we reach the desired branching factor.
                    while splits.len() < branching_factor {
                        let (largest_partition_index, _) = splits
                            .iter()
                            .enumerate()
                            .max_by_key(|&(_, (items, _))| items.len())
                            .unwrap_or_else(|| unreachable!("splits is non-empty"));
                        let (items, span) = splits.swap_remove(largest_partition_index);
                        if items.len() <= 2 {
                            // Cannot split further.
                            splits.push((items, span));
                            break;
                        }
                        let (split, _) = par_split_by_poles(items, metric, None);
                        splits.extend(split);
                    }
                } else {
                    // Use the SRF to continue splitting until the span reduction criterion is met.
                    let max_child_span = strategy.span_reduction.max_child_span_for(self.span);
                    let mut splits_heap = SizedHeap::from_iter(splits);
                    while splits_heap.peek().map_or_else(
                        || unreachable!("splits is non-empty"),
                        |(_, &span)| span > max_child_span,
                    ) {
                        // Further split the largest partition until all partitions meet the span reduction criterion.
                        let (items, span) = splits_heap.pop().unwrap_or_else(|| unreachable!("splits is non-empty"));
                        if items.len() <= 2 {
                            // Cannot split further.
                            splits_heap.push((items, span));
                            break;
                        }
                        let (split, _) = par_split_by_poles(items, metric, None);
                        splits_heap.extend(split);
                    }
                    splits = splits_heap.take_items().collect();
                }

                // Recursively create children and set the contents to the new children.
                self.contents = Contents::Children(
                    splits
                        .into_par_iter()
                        .map(|(child_items, _)| Self::par_with_center_only(child_items, metric))
                        .map(|child| child.par_partition(metric, strategy))
                        .map(Box::new)
                        .collect(),
                );
            }
            Contents::Children(children) => {
                // Replace contents so that we can check the partition criteria without running into borrow issues.
                self.contents = Contents::Children(children);

                if !(strategy.predicate)(&self) {
                    // Criteria not satisfied; convert back to a leaf by collecting all items from subtree.
                    self.contents = Contents::Leaf(self.take_subtree_items());
                }

                // Criteria are satisfied, so we continue checking children.
                // This is necessary because the user may have provided different criteria when the tree was last partitioned.

                // Take ownership of the children for partitioning.
                let children = match core::mem::replace(&mut self.contents, Contents::Leaf(Vec::new())) {
                    Contents::Children(children) => children,
                    Contents::Leaf(_) => unreachable!("We just replaced contents with children"),
                };

                // Recursively partition children and set the contents to the new children.
                let children = children
                    .into_par_iter()
                    .map(|child| child.par_partition(metric, strategy))
                    .map(Box::new)
                    .collect::<Vec<_>>();
                self.contents = Contents::Children(children);
            }
        }

        // Return the (possibly) partitioned cluster.
        self
    }
}

/// Return the number of samples to take from the given population size so as to achieve linear time complexity for geometric median estimation.
///
/// The number of samples is aggregated as follows:
///
/// - The first `sqrt_thresh` samples are taken from the population.
/// - Of the next `log2_thresh - sqrt_thresh` samples, the square root of
///   the number of samples is taken.
/// - For any remaining samples, the logarithm (base 2) of the number of
///   samples is taken.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn num_samples(population_size: usize, sqrt_thresh: usize, log2_thresh: usize) -> usize {
    if population_size < sqrt_thresh {
        population_size
    } else {
        sqrt_thresh
            + if population_size < sqrt_thresh + log2_thresh {
                ((population_size - sqrt_thresh) as f64).sqrt()
            } else {
                (log2_thresh as f64).sqrt() + ((population_size - sqrt_thresh - log2_thresh) as f64).log2()
            } as usize
    }
}

/// Splits items into two groups based on their distances to two poles.
///
/// The caller is responsible for ensuring that `items` has at least two elements.
///
/// # Arguments
///
/// * `items` - A vector of items to be split.
/// * `metric` - A function that computes the distance between two items.
/// * `arg_left` - Optional index of the left pole in `items`. If `None`, this function will find one.
///
/// # Returns
///
/// The `([(left_items, left_span), (right_items, right_span)], span)` where:
///   - `left_items` are items closer to the left pole (including the left pole itself),
///   - `right_items` are items closer to the right pole (including the right pole itself),
///   - `left_span` is the distance from the left pole to the farthest item in `left_items`,
///   - `right_span` is the distance from the right pole to the farthest item in `right_items`.
///   - `span` is the distance between the two poles.
#[expect(clippy::type_complexity)]
fn split_by_poles<Id, I, T: DistanceValue, M: Fn(&I, &I) -> T>(
    mut items: Vec<(Id, I)>,
    metric: &M,
    arg_left: Option<usize>,
) -> ([(Vec<(Id, I)>, T); 2], T) {
    let left_pole = if let Some(arg_left) = arg_left {
        items.swap_remove(arg_left)
    } else {
        let maybe_pole = items.pop().unwrap_or_else(|| unreachable!("items is non-empty"));

        let mut maybe_distances = items
            .into_iter()
            .map(|item| (metric(&maybe_pole.1, &item.1), item))
            .collect::<Vec<_>>();

        let arg_left = maybe_distances
            .iter()
            .enumerate()
            .max_by_key(|&(i, &(d, _))| crate::utils::MaxItem(i, d))
            .map_or(0, |(i, _)| i);
        let (_, left_pole) = maybe_distances.swap_remove(arg_left);

        items = maybe_distances.into_iter().map(|(_, item)| item).collect();
        items.push(maybe_pole);
        left_pole
    };

    // Compute distances from left pole to all other items, and keep the distances with their respective items for later.
    let mut left_distances = items
        .into_iter()
        .map(|item| (metric(&left_pole.1, &item.1), item))
        .collect::<Vec<_>>();

    // The right pole is the farthest item from the left pole
    let (arg_right, &(span, _)) = left_distances
        .iter()
        .enumerate()
        .max_by_key(|&(i, &(d, _))| crate::utils::MaxItem(i, d))
        .unwrap_or_else(|| unreachable!("left_distances is non-empty"));
    // Remove it from the items list.
    let right_pole = left_distances.swap_remove(arg_right).1;

    // At this point, we have two poles, and an item list which does not contain the poles.

    // Compute distances from right pole to all items and partition items based on which pole they are closer to. Ties go to the left pole.
    let (left_assigned, right_assigned) = left_distances
        .into_iter()
        .map(|(l, item)| (l, metric(&right_pole.1, &item.1), item))
        .partition::<Vec<_>, _>(|&(l, r, _)| l <= r);

    // Collect items assigned to each pole, lacing the poles first, though their order does not matter.
    let (left_items, left_distances): (Vec<_>, Vec<_>) = core::iter::once((left_pole, T::zero()))
        .chain(left_assigned.into_iter().map(|(l, _, item)| (item, l)))
        .unzip();
    let (right_items, right_distances): (Vec<_>, Vec<_>) = core::iter::once((right_pole, T::zero()))
        .chain(right_assigned.into_iter().map(|(_, r, item)| (item, r)))
        .unzip();

    let left_span = left_distances
        .into_iter()
        .max_by_key(|&d| MaxItem(0, d))
        .unwrap_or_else(|| unreachable!("left_items is non-empty"));
    let right_span = right_distances
        .into_iter()
        .max_by_key(|&d| MaxItem(0, d))
        .unwrap_or_else(|| unreachable!("right_items is non-empty"));

    ([(left_items, left_span), (right_items, right_span)], span)
}

/// Parallel version of [`split_by_poles`](split_by_poles).
#[expect(clippy::type_complexity)]
fn par_split_by_poles<
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
>(
    mut items: Vec<(Id, I)>,
    metric: &M,
    arg_left: Option<usize>,
) -> ([(Vec<(Id, I)>, T); 2], T) {
    let left_pole = if let Some(arg_left) = arg_left {
        items.swap_remove(arg_left)
    } else {
        let maybe_pole = items.pop().unwrap_or_else(|| unreachable!("items is non-empty"));

        let mut maybe_distances = items
            .into_par_iter()
            .map(|item| (metric(&maybe_pole.1, &item.1), item))
            .collect::<Vec<_>>();

        let arg_left = maybe_distances
            .par_iter()
            .enumerate()
            .max_by_key(|&(i, &(d, _))| crate::utils::MaxItem(i, d))
            .map_or(0, |(i, _)| i);
        let (_, left_pole) = maybe_distances.swap_remove(arg_left);

        items = maybe_distances.into_iter().map(|(_, item)| item).collect();
        items.push(maybe_pole);
        left_pole
    };

    // Compute distances from left pole to all other items, and keep the distances with their respective items for later.
    let mut left_distances = items
        .into_par_iter()
        .map(|item| (metric(&left_pole.1, &item.1), item))
        .collect::<Vec<_>>();

    // The right pole is the farthest item from the left pole
    let (arg_right, &(span, _)) = left_distances
        .par_iter()
        .enumerate()
        .max_by_key(|&(i, &(d, _))| crate::utils::MaxItem(i, d))
        .unwrap_or_else(|| unreachable!("left_distances is non-empty"));
    // Remove it from the items list.
    let right_pole = left_distances.swap_remove(arg_right).1;

    // At this point, we have two poles, and an item list which does not contain the poles.

    // Compute distances from right pole to all items and partition items based on which pole they are closer to. Ties go to the left pole.
    let (left_assigned, right_assigned): (Vec<_>, Vec<_>) = left_distances
        .into_par_iter()
        .map(|(l, item)| (l, metric(&right_pole.1, &item.1), item))
        .partition(|&(l, r, _)| l <= r);

    // Collect items assigned to each pole, lacing the poles first, though their order does not matter.
    let (left_items, left_distances): (Vec<_>, Vec<_>) = core::iter::once((left_pole, T::zero()))
        .chain(left_assigned.into_iter().map(|(l, _, item)| (item, l)))
        .unzip();
    let (right_items, right_distances): (Vec<_>, Vec<_>) = core::iter::once((right_pole, T::zero()))
        .chain(right_assigned.into_iter().map(|(_, r, item)| (item, r)))
        .unzip();

    let left_span = left_distances
        .into_iter()
        .max_by_key(|&d| MaxItem(0, d))
        .unwrap_or_else(|| unreachable!("left_items is non-empty"));
    let right_span = right_distances
        .into_iter()
        .max_by_key(|&d| MaxItem(0, d))
        .unwrap_or_else(|| unreachable!("right_items is non-empty"));

    ([(left_items, left_span), (right_items, right_span)], span)
}
