//! The most basic representation of a `Cluster` is a metric-`Ball`.

use core::fmt::Debug;

use rayon::prelude::*;

use crate::{utils, DistanceValue};

/// A `Ball` is a collection of items in a dataset that are within a `radius` from a `center` item.
///
/// # Type Parameters
///
/// * `I`: The type of items in the tree.
/// * `Id`: The type of metadata associated each item.
/// * `T`: The type of distance values between items.
/// * `A`: The type of arbitrary data associated with the ball.
#[must_use]
pub struct Ball<Id, I, T: DistanceValue, A> {
    /// The number of items in the ball, including the center.
    cardinality: usize,
    /// The center item of the ball.
    center: (Id, I),
    /// The radius of the ball.
    radius: T,
    /// The Local Fractal Dimension (LFD) of the ball.
    lfd: f64,
    /// The sum of all radial distances from the center to all items in the ball.
    radial_sum: T,
    /// The `Contents` of the ball.
    contents: Contents<Id, I, T, A>,
    /// Arbitrary data associated with the ball.
    annotation: Option<A>,
}

/// The contents of a `Ball` can either be a collection of items (if it is a leaf) or a collection of child `Ball`s (if it is a parent).
enum Contents<Id, I, T: DistanceValue, A> {
    /// The ball is a leaf and contains items directly.
    Leaf(Vec<(Id, I)>),
    /// The ball is a parent and contains child balls.
    Children([Box<Ball<Id, I, T, A>>; 2]),
}

impl<I: Debug, Id: Debug, T: DistanceValue + Debug, A: Debug> Debug for Ball<Id, I, T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ball")
            .field("cardinality", &self.cardinality)
            .field("center", &self.center)
            .field("radius", &self.radius)
            .field("lfd", &self.lfd)
            .field("radial_sum", &self.radial_sum)
            .field("contents", &self.contents)
            .field("annotation", &self.annotation)
            .finish()
    }
}

impl<Id, I, T: DistanceValue + Debug, A: Debug> Debug for Contents<Id, I, T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Leaf(items) => f.debug_tuple("Leaf").field(&items.len()).finish(),
            Self::Children(_) => f.debug_tuple("Children").finish(),
        }
    }
}

impl<I, T: DistanceValue, A> Ball<usize, I, T, A> {
    /// Create a new tree of `Ball`s with `usize` indices as item metadata.
    ///
    /// # Errors
    ///
    /// - See [`new_tree`](Self::new_tree) for details.
    pub fn new_tree_with_indices<M: Fn(&I, &I) -> T>(
        items: Vec<I>,
        metric: &M,
        criteria: &impl Fn(&Self) -> bool,
    ) -> Result<Self, String> {
        let indexed_items = items.into_iter().enumerate().collect();
        Self::new_tree(indexed_items, metric, criteria)
    }
}

impl<I: Send + Sync, T: DistanceValue + Send + Sync, A: Send + Sync> Ball<usize, I, T, A> {
    /// Parallel version of [`new_tree_with_indices`](Self::new_tree_with_indices).
    ///
    /// # Errors
    ///
    /// - See [`new_tree`](Self::new_tree) for details.
    pub fn par_new_tree_with_indices<M: Fn(&I, &I) -> T + Send + Sync>(
        items: Vec<I>,
        metric: &M,
        criteria: &(impl Fn(&Self) -> bool + Send + Sync),
    ) -> Result<Self, String> {
        let indexed_items = items.into_iter().enumerate().collect();
        Self::par_new_tree(indexed_items, metric, criteria)
    }
}

impl<Id, I, T: DistanceValue, A> Ball<Id, I, T, A> {
    /// The number of items in the ball, including the center.
    pub const fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Returns a reference to the id of the center item of the ball.
    pub const fn center_id(&self) -> &Id {
        &self.center.0
    }

    /// A reference to the center item of the ball.
    pub const fn center(&self) -> &I {
        &self.center.1
    }

    /// The radius of the ball.
    pub const fn radius(&self) -> T {
        self.radius
    }

    /// The Local Fractal Dimension (LFD) of the ball.
    pub const fn lfd(&self) -> f64 {
        self.lfd
    }

    /// The sum of all radial distances from the center to all items in the ball.
    pub const fn radial_sum(&self) -> T {
        self.radial_sum
    }

    /// A reference to the arbitrary data associated with the ball, if any.
    pub const fn annotation(&self) -> Option<&A> {
        self.annotation.as_ref()
    }

    /// A vector of references to all clusters in the tree in pre-order (i.e., parent before children).
    pub fn subtree(&self) -> Vec<&Self> {
        match &self.contents {
            Contents::Leaf(_) => vec![self],
            Contents::Children([left, right]) => core::iter::once(self)
                .chain(left.subtree())
                .chain(right.subtree())
                .collect(),
        }
    }

    /// A vector of references to all items in the subtree rooted at this ball,
    /// excluding the center of this ball.
    pub fn subtree_items(&self) -> Vec<&(Id, I)> {
        match &self.contents {
            Contents::Leaf(items) => items.iter().collect(),
            Contents::Children([left, right]) => left.all_items().into_iter().chain(right.all_items()).collect(),
        }
    }

    /// A vector of references to all items in the ball, including the center, which is placed first.
    pub fn all_items(&self) -> Vec<&(Id, I)> {
        match &self.contents {
            Contents::Leaf(items) => core::iter::once(&self.center).chain(items.iter()).collect(),
            Contents::Children([left, right]) => core::iter::once(&self.center)
                .chain(left.all_items())
                .chain(right.all_items())
                .collect(),
        }
    }

    /// A vector of references to the child balls of this ball. Returns `None` if the ball is a leaf.
    pub fn children(&self) -> Option<[&Self; 2]> {
        match &self.contents {
            Contents::Leaf(_) => None,
            Contents::Children([left, right]) => Some([left, right]),
        }
    }

    /// Checks if the ball is a leaf.
    pub const fn is_leaf(&self) -> bool {
        matches!(self.contents, Contents::Leaf(_))
    }

    /// Checks if the ball is a singleton (i.e., contains only one distinct item).
    pub fn is_singleton(&self) -> bool {
        self.cardinality == 1 || self.radius.is_zero()
    }

    /// Returns the distance from the given item to the center of the ball using the provided metric.
    pub fn distance_to_center<M: Fn(&I, &I) -> T>(&self, item: &I, metric: &M) -> (&(Id, I), T) {
        (&self.center, metric(item, &self.center.1))
    }

    /// Returns the distance from the given item to all items in the ball and its subtree using the provided metric.
    pub fn distances_to_all_items<M: Fn(&I, &I) -> T>(&self, item: &I, metric: &M) -> Vec<(&Id, &I, T)> {
        self.all_items().iter().map(|(i, p)| (i, p, metric(item, p))).collect()
    }

    /// Creates a new tree of `Ball`s.
    ///
    /// # Parameters
    ///
    /// * `items`: The items to be clustered into a tree of balls.
    /// * `metric`: A function that computes the distance between two items.
    /// * `criteria`: A function that determines whether a ball should be partitioned into child balls. As a default, the user can use `&|_| true`.
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
            return Err("Cannot create a Ball tree with no items".to_string());
        }
        let criteria = |b: &Self| !b.is_singleton() && criteria(b);
        Ok(Self::with_center_only(items, metric).partition(metric, &criteria))
    }

    /// Annotates all balls in the tree by applying the provided functions.
    ///
    /// # Parameters
    ///
    /// * `pre`: A function that computes a pre-order annotation for a ball. It is applied before the children are annotated.
    /// * `post`: A function that computes a post-order annotation for a ball. It is applied after the children are annotated.
    /// * `metric`: A function that computes the distance between two items.
    pub fn annotate<M: Fn(&I, &I) -> T, Pre: Fn(&Self, &M) -> Option<A>, Post: Fn(&Self, &M) -> Option<A>>(
        &mut self,
        pre: &Pre,
        post: &Post,
        metric: &M,
    ) {
        self.annotation = pre(self, metric);
        if let Contents::Children([left, right]) = &mut self.contents {
            left.annotate(pre, post, metric);
            right.annotate(pre, post, metric);
        }
        self.annotation = post(self, metric);
    }

    /// Removes all annotations from the balls in the tree.
    pub fn clear_annotations(&mut self) {
        self.annotation = None;
        if let Contents::Children([left, right]) = &mut self.contents {
            left.clear_annotations();
            right.clear_annotations();
        }
    }

    /// Traverses the tree in pre-order, checking the provided predicate on each ball, and converts balls that satisfy the predicate into leaves by collecting
    /// all items from their descendants and dropping the descendants in the process.
    pub fn prune<P: Fn(&Self) -> bool>(&mut self, predicate: &P) {
        if predicate(self) {
            // The predicate is satisfied, so we convert this ball to a leaf by collecting all items from its descendants.
            self.contents = Contents::Leaf(self.take_subtree_items());
        } else if let Contents::Children([left, right]) = &mut self.contents {
            // The predicate is not satisfied, so we continue checking children.
            left.prune(predicate);
            right.prune(predicate);
        }
    }

    /// Private constructor for `Ball`.
    ///
    /// WARNING: This function does only sets the `center` and `cardinality` fields correctly. Other fields are placeholders and must be computed in
    /// `partition`.
    fn with_center_only<M: Fn(&I, &I) -> T>(mut items: Vec<(Id, I)>, metric: &M) -> Self {
        if items.len() == 1 {
            // A singleton ball: `center` is the only item, `radius` is 0, `LFD` is 1
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
                    let num_samples = utils::num_samples(items.len(), 100, 10_000);
                    &items[..num_samples]
                };
                // Remove and return the `center`.
                let gm_index = geometric_median(gm_sample, metric);
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

    /// Partitions the ball into two child balls based on the provided `metric` and `criteria`, then recursively partitions the children until the criteria are
    /// no longer satisfied.
    ///
    /// # Parameters
    ///
    /// * `metric`: A function that computes the distance between two items.
    /// * `criteria`: A function that determines whether a ball should be partitioned into child balls. As a default, the user can use `&|_| true`.
    pub fn partition<M: Fn(&I, &I) -> T>(mut self, metric: &M, criteria: &impl Fn(&Self) -> bool) -> Self {
        match self.contents {
            Contents::Leaf(items) => {
                if items.is_empty() {
                    // A singleton ball: nothing to partition.
                    self.radius = T::zero();
                    self.lfd = 1.0; // LFD of a singleton is _defined_ as 1
                    self.contents = Contents::Leaf(Vec::new());
                    return self;
                }
                if items.len() == 1 {
                    // A ball with one center and one item: nothing to partition.
                    self.radius = metric(&self.center.1, &items[0].1);
                    self.lfd = 1.0; // LFD of clusters with 2 items is _defined_ as 1
                    self.radial_sum = self.radius;
                    self.contents = Contents::Leaf(items);
                    return self;
                }

                // At this point, we have at least 2 items so radius computation is meaningful, and we can always remove two poles to partition with.
                // We have to first compute the radius and LFD because `with_center_only` does not compute them.

                // Compute the radius and LFD of the ball.
                let radial_distances = items
                    .iter()
                    .map(|item| metric(&self.center.1, &item.1))
                    .collect::<Vec<_>>();
                let arg_radius = radial_distances
                    .iter()
                    .enumerate()
                    .max_by_key(|&(i, &d)| utils::MaxItem(i, d))
                    .map_or(0, |(i, _)| i);
                self.radius = radial_distances[arg_radius];
                self.lfd = lfd_estimate(&radial_distances, self.radius);
                self.radial_sum = radial_distances.iter().copied().sum();

                // Replace contents so that we can check the partition criteria without running into borrow issues.
                self.contents = Contents::Leaf(items);

                if !criteria(&self) {
                    // Criteria not satisfied; do not partition.
                    return self;
                }

                // Criteria are satisfied, so we partition the ball.

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
                    .max_by_key(|&(i, &(d, _))| utils::MaxItem(i, d))
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

        // Return the (possibly) partitioned ball.
        self
    }

    /// Removes and returns all items from the ball and its descendants, excluding the center of this ball; the children are dropped in the process and this
    /// ball becomes a leaf with no items other than its center.
    pub fn take_subtree_items(&mut self) -> Vec<(Id, I)> {
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

impl<Id: Send + Sync, I: Send + Sync, T: DistanceValue + Send + Sync, A: Send + Sync> Ball<Id, I, T, A> {
    /// Parallel version of [`distance_to_all`](Self::distances_to_all).
    pub fn par_distances_to_all_items<M: Fn(&I, &I) -> T + Send + Sync>(
        &self,
        item: &I,
        metric: &M,
    ) -> Vec<(&Id, &I, T)> {
        self.all_items()
            .par_iter()
            .map(|(id, p)| (id, p, metric(item, p)))
            .collect()
    }

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
            return Err("Cannot create a Ball tree with no items".to_string());
        }
        let criteria = |b: &Self| !b.is_singleton() && criteria(b);
        Ok(Self::par_with_center_only(items, metric).par_partition(metric, &criteria))
    }

    /// Parallel version of [`annotate`](Self::annotate).
    pub fn par_annotate<
        M: Fn(&I, &I) -> T + Send + Sync,
        Pre: Fn(&Self, &M) -> Option<A> + Send + Sync,
        Post: Fn(&Self, &M) -> Option<A> + Send + Sync,
    >(
        &mut self,
        pre: &Pre,
        post: &Post,
        metric: &M,
    ) {
        self.annotation = pre(self, metric);
        if let Contents::Children([left, right]) = &mut self.contents {
            rayon::join(
                || left.par_annotate(pre, post, metric),
                || right.par_annotate(pre, post, metric),
            );
        }
        self.annotation = post(self, metric);
    }

    /// Parallel version of [`clear_annotations`](Self::clear_annotations).
    pub fn par_clear_annotations(&mut self) {
        self.annotation = None;
        if let Contents::Children([left, right]) = &mut self.contents {
            rayon::join(|| left.par_clear_annotations(), || right.par_clear_annotations());
        }
    }

    /// Parallel version of [`prune`](Self::prune).
    pub fn par_prune<P: (Fn(&Self) -> bool) + Send + Sync>(&mut self, predicate: &P) {
        if predicate(self) {
            // The predicate is satisfied, so we convert this ball to a leaf by collecting all items from its subtree.
            self.contents = Contents::Leaf(self.take_subtree_items());
        } else if let Contents::Children([left, right]) = &mut self.contents {
            // The predicate is not satisfied, so we continue checking children.
            rayon::join(|| left.par_prune(predicate), || right.par_prune(predicate));
        }
    }

    /// Parallel version of [`with_center_only`](Self::with_center_only).
    fn par_with_center_only<M: Fn(&I, &I) -> T + Send + Sync>(mut items: Vec<(Id, I)>, metric: &M) -> Self {
        if items.len() == 1 {
            // A singleton ball: center is the only item, radius is 0, LFD is 1
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
                    let num_samples = utils::num_samples(items.len(), 100, 10_000);
                    &items[..num_samples]
                };
                let gm_index = par_geometric_median(gm_sample, metric);
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
                    // A singleton ball: nothing to partition.
                    self.radius = T::zero();
                    self.lfd = 1.0; // LFD of a singleton is _defined_ as 1
                    self.contents = Contents::Leaf(Vec::new());
                    return self;
                }
                if items.len() == 1 {
                    // A ball with one center and one item: nothing to partition.
                    self.radius = metric(&self.center.1, &items[0].1);
                    self.lfd = 1.0; // LFD of clusters with 2 items is _defined_ as 1
                    self.radial_sum = self.radius;
                    self.contents = Contents::Leaf(items);
                    return self;
                }

                // At this point, we have at least 2 items so radius computation is meaningful, and we can always remove two poles to partition with.
                // We have to first compute the radius and LFD because `new` does not compute them.

                // Compute the radius and LFD of the ball.
                let radial_distances = items
                    .par_iter()
                    .map(|item| metric(&self.center.1, &item.1))
                    .collect::<Vec<_>>();
                let arg_radius = radial_distances
                    .par_iter()
                    .enumerate()
                    .max_by_key(|&(i, &d)| utils::MaxItem(i, d))
                    .map_or(0, |(i, _)| i);
                self.radius = radial_distances[arg_radius];
                self.lfd = lfd_estimate(&radial_distances, self.radius);
                self.radial_sum = radial_distances.par_iter().copied().sum();

                // Replace contents so that we can check the partition criteria without running into borrow issues.
                self.contents = Contents::Leaf(items);

                if !criteria(&self) {
                    // Criteria not satisfied; do not partition.
                    return self;
                }

                // Criteria are satisfied, so we partition the ball.

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
                    .max_by_key(|&(i, &(d, _))| utils::MaxItem(i, d))
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

        // Return the (possibly) partitioned ball.
        self
    }
}

/// Estimates the Local Fractal Dimension (LFD) of a ball given the distances
/// of its items from the center and the radius of the ball.
///
/// This uses the formula `log2(N / n)`, where `N` is the total number of items
/// in the ball, and `n` is the number of items within half the radius.
///
/// If the radius is zero or if there are no items within half the radius,
/// the LFD is defined to be 1.0.
#[expect(clippy::cast_precision_loss)]
pub fn lfd_estimate<T: DistanceValue>(distances: &[T], radius: T) -> f64 {
    let half_radius = radius.to_f64().unwrap_or(0.0) / 2.0;
    if distances.is_empty() || distances.len() == 1 || half_radius <= f64::EPSILON {
        // In all three of the following cases, we define LFD to be 1.0:
        //   - No non-center items (singleton ball)
        //   - One non-center item (ball with two items)
        //   - Radius is zero or too small to be meaningful
        1.0
    } else {
        // The ball has at least 2 non-center items, so LFD computation is
        // meaningful.

        // Count how many items are within half the radius.
        // We use f64::MAX as a sentinel to exclude items whose distance
        // could not be converted to f64.
        let count = distances
            .iter()
            .map(|d| d.to_f64().unwrap_or(f64::MAX))
            .filter(|&d| d <= half_radius)
            .count()
            + 1; // +1 to include the center

        // Compute and return the LFD. This is well-defined because
        // `distances.len() >= 2` and `count >= 1`, so the argument to log2
        // is always >= 1.0
        ((distances.len() as f64) / (count as f64)).log2()
    }
}

/// Returns the index of the geometric median of the given items.
///
/// The geometric median is the item that minimizes the sum of distances to
/// all other items in the slice.
///
/// The user must ensure that the items slice is not empty.
fn geometric_median<I, Id, T: DistanceValue, M: Fn(&I, &I) -> T>(items: &[(Id, I)], metric: &M) -> usize {
    // Compute the full distance matrix for the items.
    let distance_matrix = {
        let mut matrix = vec![vec![T::zero(); items.len()]; items.len()];
        for (r, (_, i)) in items.iter().enumerate() {
            for (c, (_, j)) in items.iter().enumerate().take(r) {
                let d = metric(i, j);
                matrix[r][c] = d;
                matrix[c][r] = d;
            }
        }
        matrix
    };

    // Find the index of the item with the minimum total distance to all other items.
    distance_matrix
        .into_iter()
        .map(|row| row.into_iter().sum::<T>())
        .enumerate()
        .min_by_key(|&(i, v)| utils::MinItem(i, v))
        .map_or_else(|| unreachable!("items must be non-empty"), |(i, _)| i)
}

/// Parallel version of [`geometric_median`](geometric_median).
fn par_geometric_median<
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    M: (Fn(&I, &I) -> T) + Send + Sync,
>(
    items: &[(Id, I)],
    metric: &M,
) -> usize {
    // Compute the full distance matrix for the items in parallel.
    let distance_matrix = {
        let matrix = vec![vec![T::zero(); items.len()]; items.len()];
        items.par_iter().enumerate().for_each(|(r, (_, i))| {
            items.par_iter().enumerate().take(r).for_each(|(c, (_, j))| {
                let d = metric(i, j);
                // SAFETY: We have exclusive access to each cell in the matrix
                // because every (r, c) pair is unique.
                #[allow(unsafe_code)]
                unsafe {
                    let row_ptr = &mut *matrix.as_ptr().cast_mut().add(r);
                    row_ptr[c] = d;

                    let col_ptr = &mut *matrix.as_ptr().cast_mut().add(c);
                    col_ptr[r] = d;
                }
            });
        });
        matrix
    };

    // Find the index of the item with the minimum total distance to all
    // other items.
    distance_matrix
        .into_par_iter()
        .map(|row| row.into_iter().sum::<T>())
        .enumerate()
        .min_by_key(|&(i, v)| utils::MinItem(i, v))
        .map_or_else(|| unreachable!("items must be non-empty"), |(i, _)| i)
}
