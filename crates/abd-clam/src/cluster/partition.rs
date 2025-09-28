//! A `Cluster` can be partitioned into a tree.

use rayon::prelude::*;

use crate::DistanceValue;

use super::{Cluster, Contents};

impl<Id, I, T: DistanceValue, A> Cluster<Id, I, T, A> {
    /// Creates a new tree of `Cluster`s.
    ///
    /// # Parameters
    ///
    /// * `items`: The items to be clustered into a tree of clusters.
    /// * `metric`: A function that computes the distance between two items.
    /// * `criteria`: A function that determines whether a cluster should be partitioned into child clusters. As a default, the user can use `&|_| true`.
    ///
    /// # Errors
    ///
    /// - If `items` is empty.
    pub fn new_tree<M: Fn(&I, &I) -> T>(
        items: Vec<(Id, I)>,
        metric: &M,
        criteria: &impl Fn(&Self) -> bool,
    ) -> Result<Self, String> {
        if items.is_empty() {
            return Err("Cannot create a Cluster tree with no items".to_string());
        }
        let criteria = |b: &Self| !b.is_singleton() && criteria(b);
        Ok(Self::with_center_only(items, metric).partition(metric, &criteria))
    }

    /// Private constructor for `Cluster`.
    ///
    /// WARNING: This function does only sets the `center` and `cardinality` fields correctly. Other fields are placeholders and must be computed in
    /// `partition`.
    fn with_center_only<M: Fn(&I, &I) -> T>(mut items: Vec<(Id, I)>, metric: &M) -> Self {
        if items.len() == 1 {
            // A singleton cluster: `center` is the only item, `radius` is 0, `LFD` is 1
            let center = items.pop().unwrap_or_else(|| unreachable!("Cardinality is 1"));
            Self {
                cardinality: 1,
                center,
                radius: T::zero(),
                lfd: 1.0, // LFD of a singleton is _defined_ as 1
                radial_sum: T::zero(),
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
                contents: Contents::Leaf(items),
                annotation: None,
            }
        }
    }

    /// Partitions the cluster into two child clusters based on the provided `metric` and `criteria`, then recursively partitions the children until the criteria are
    /// no longer satisfied.
    ///
    /// # Parameters
    ///
    /// * `metric`: A function that computes the distance between two items.
    /// * `criteria`: A function that determines whether a cluster should be partitioned into child clusters. As a default, the user can use `&|_| true`.
    pub fn partition<M: Fn(&I, &I) -> T>(mut self, metric: &M, criteria: &impl Fn(&Self) -> bool) -> Self {
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
                let arg_radius = radial_distances
                    .iter()
                    .enumerate()
                    .max_by_key(|&(i, &d)| crate::utils::MaxItem(i, d))
                    .map_or(0, |(i, _)| i);
                self.radius = radial_distances[arg_radius];
                self.lfd = crate::utils::lfd_estimate(&radial_distances, self.radius);
                self.radial_sum = radial_distances.iter().copied().sum();

                // Replace contents so that we can check the partition criteria without running into borrow issues.
                self.contents = Contents::Leaf(items);

                if !criteria(&self) {
                    // Criteria not satisfied; do not partition.
                    return self;
                }

                // Criteria are satisfied, so we partition the cluster.

                // Take ownership of the items for partitioning.
                let mut items = match core::mem::replace(&mut self.contents, Contents::Leaf(Vec::new())) {
                    Contents::Leaf(items) => items,
                    Contents::Children(_) => unreachable!("We just replaced contents with a leaf"),
                };

                // The left pole is the farthest item from the center. Remove it from the items list.
                let left_pole = items.swap_remove(arg_radius);

                // Compute distances from left pole to all other items, and keep the distances with their respective items for later.
                let mut left_distances = items
                    .into_iter()
                    .map(|item| (metric(&left_pole.1, &item.1), item))
                    .collect::<Vec<_>>();

                // The right pole is the farthest item from the left pole
                let arg_right = left_distances
                    .iter()
                    .enumerate()
                    .max_by_key(|&(i, &(d, _))| crate::utils::MaxItem(i, d))
                    .map_or(0, |(i, _)| i);
                // Remove it from the items list.
                let right_pole = left_distances.swap_remove(arg_right).1;

                // At this point, we have two poles, and an item list which does not contain the poles.

                // Compute distances from right pole to all items and partition items based on which pole they are closer to. Ties go to the left pole.
                let (left_assigned, right_assigned) = left_distances
                    .into_iter()
                    .map(|(l, item)| (l, metric(&right_pole.1, &item.1), item))
                    .partition::<Vec<_>, _>(|&(l, r, _)| l <= r);

                // Collect items assigned to each pole, lacing the poles first, though their order does not matter.
                let left_items = core::iter::once(left_pole)
                    .chain(left_assigned.into_iter().map(|(_, _, item)| item))
                    .collect::<Vec<_>>();
                let right_items = core::iter::once(right_pole)
                    .chain(right_assigned.into_iter().map(|(_, _, item)| item))
                    .collect::<Vec<_>>();

                // Recursively create children and set the contents to the new children.
                self.contents = Contents::Children([
                    Box::new(Self::with_center_only(left_items, metric).partition(metric, criteria)),
                    Box::new(Self::with_center_only(right_items, metric).partition(metric, criteria)),
                ]);
            }
            Contents::Children(children) => {
                // Replace contents so that we can check the partition criteria without running into borrow issues.
                self.contents = Contents::Children(children);

                if !criteria(&self) {
                    // Criteria not satisfied; convert back to a leaf by collecting all items from subtree.
                    self.contents = Contents::Leaf(self.take_subtree_items());
                }

                // Criteria are satisfied, so we continue checking children.
                // This is necessary because the user may have provided different criteria when the tree was last partitioned.

                // Take ownership of the children for partitioning.
                let [left, right] = match core::mem::replace(&mut self.contents, Contents::Leaf(Vec::new())) {
                    Contents::Children(children) => children,
                    Contents::Leaf(_) => unreachable!("We just replaced contents with children"),
                };

                // Recursively partition children and set the contents to the new children.
                self.contents = Contents::Children([
                    Box::new(left.partition(metric, criteria)),
                    Box::new(right.partition(metric, criteria)),
                ]);
            }
        }

        // Return the (possibly) partitioned cluster.
        self
    }

    /// Removes and returns all items from the cluster and its descendants, excluding the center of this cluster; the children are dropped in the process and this
    /// cluster becomes a leaf with no items other than its center.
    ///
    /// WARNING: This function does not recompute any properties for the up to the root after removing the items. The caller must ensure that the properties are
    /// still valid after this operation.
    pub(crate) fn take_subtree_items(&mut self) -> Vec<(Id, I)> {
        // Take ownership of the contents so we can recurse and drop children.
        let contents = core::mem::replace(&mut self.contents, Contents::Leaf(Vec::new()));
        match contents {
            Contents::Leaf(items) => items,
            Contents::Children([mut left, mut right]) => {
                let mut items = Vec::with_capacity(self.cardinality - 1);
                items.extend(left.take_subtree_items());
                items.push(left.center);

                items.extend(right.take_subtree_items());
                items.push(right.center);

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
    pub fn par_new_tree<M: Fn(&I, &I) -> T + Send + Sync>(
        items: Vec<(Id, I)>,
        metric: &M,
        criteria: &(impl Fn(&Self) -> bool + Send + Sync),
    ) -> Result<Self, String> {
        if items.is_empty() {
            return Err("Cannot create a Cluster tree with no items".to_string());
        }
        let criteria = |b: &Self| !b.is_singleton() && criteria(b);
        Ok(Self::par_with_center_only(items, metric).par_partition(metric, &criteria))
    }

    /// Parallel version of [`with_center_only`](Self::with_center_only).
    fn par_with_center_only<M: Fn(&I, &I) -> T + Send + Sync>(mut items: Vec<(Id, I)>, metric: &M) -> Self {
        if items.len() == 1 {
            // A singleton cluster: center is the only item, radius is 0, LFD is 1
            let center = items.pop().unwrap_or_else(|| unreachable!("Cardinality is 1"));
            Self {
                cardinality: 1,
                center,
                radius: T::zero(),
                lfd: 1.0, // LFD of a singleton is _defined_ as 1
                radial_sum: T::zero(),
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
                contents: Contents::Leaf(items),
                annotation: None,
            }
        }
    }

    /// Parallel version of [`partition`](Self::partition).
    pub fn par_partition<M: Fn(&I, &I) -> T + Send + Sync>(
        mut self,
        metric: &M,
        criteria: &(impl Fn(&Self) -> bool + Send + Sync),
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
                let arg_radius = radial_distances
                    .par_iter()
                    .enumerate()
                    .max_by_key(|&(i, &d)| crate::utils::MaxItem(i, d))
                    .map_or(0, |(i, _)| i);
                self.radius = radial_distances[arg_radius];
                self.lfd = crate::utils::lfd_estimate(&radial_distances, self.radius);
                self.radial_sum = radial_distances.par_iter().copied().sum();

                // Replace contents so that we can check the partition criteria without running into borrow issues.
                self.contents = Contents::Leaf(items);

                if !criteria(&self) {
                    // Criteria not satisfied; do not partition.
                    return self;
                }

                // Criteria are satisfied, so we partition the cluster.

                // Take ownership of the items for partitioning.
                let mut items = match core::mem::replace(&mut self.contents, Contents::Leaf(Vec::new())) {
                    Contents::Leaf(items) => items,
                    Contents::Children(_) => unreachable!("We just replaced contents with a leaf"),
                };

                // The left pole is the farthest item from the center. Remove it from the items list.
                let left_pole = items.swap_remove(arg_radius);

                // Compute distances from left pole to all other items, and keep the distances with their respective items for later.
                let mut left_distances = items
                    .into_par_iter()
                    .map(|item| (metric(&left_pole.1, &item.1), item))
                    .collect::<Vec<_>>();

                // The right pole is the farthest item from the left pole
                let arg_right = left_distances
                    .par_iter()
                    .enumerate()
                    .max_by_key(|&(i, &(d, _))| crate::utils::MaxItem(i, d))
                    .map_or(0, |(i, _)| i);
                // Remove it from the items list.
                let right_pole = left_distances.swap_remove(arg_right).1;

                // At this point, we have two poles, and an item list which does not contain the poles.

                // Compute distances from right pole to all items and partition items based on which pole they are closer to. Ties go to the left pole.
                let (left_assigned, right_assigned): (Vec<_>, Vec<_>) = left_distances
                    .into_par_iter()
                    .map(|(l, item)| (l, metric(&right_pole.1, &item.1), item))
                    .partition(|&(l, r, _)| l <= r);

                // Collect items assigned to each pole, lacing the poles first, though their order does not matter.
                let left_items = core::iter::once(left_pole)
                    .chain(left_assigned.into_iter().map(|(_, _, item)| item))
                    .collect::<Vec<_>>();
                let right_items = core::iter::once(right_pole)
                    .chain(right_assigned.into_iter().map(|(_, _, item)| item))
                    .collect::<Vec<_>>();

                // Recursively create children and set the contents to the new children.
                let (left, right) = rayon::join(
                    || Self::par_with_center_only(left_items, metric).par_partition(metric, criteria),
                    || Self::par_with_center_only(right_items, metric).par_partition(metric, criteria),
                );
                self.contents = Contents::Children([Box::new(left), Box::new(right)]);
            }
            Contents::Children(children) => {
                // Replace contents so that we can check the partition criteria without running into borrow issues.
                self.contents = Contents::Children(children);

                if !criteria(&self) {
                    // Criteria not satisfied; convert back to a leaf by collecting all items from subtree.
                    self.contents = Contents::Leaf(self.take_subtree_items());
                }

                // Criteria are satisfied, so we continue checking children.
                // This is necessary because the user may have provided different criteria when the tree was last partitioned.

                // Take ownership of the children for partitioning.
                let [left, right] = match core::mem::replace(&mut self.contents, Contents::Leaf(Vec::new())) {
                    Contents::Children(children) => children,
                    Contents::Leaf(_) => unreachable!("We just replaced contents with children"),
                };

                // Recursively partition children and set the contents to the new children.
                let (left, right) = rayon::join(
                    || left.par_partition(metric, criteria),
                    || right.par_partition(metric, criteria),
                );
                self.contents = Contents::Children([Box::new(left), Box::new(right)]);
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
