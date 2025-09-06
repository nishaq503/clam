//! Partitioning algorithms for the tree and clusters.

use crate::{
    Cluster, DistanceValue,
    utils::{geometric_median, lfd_estimate},
};

mod par_partition;
pub mod strategy;

use strategy::PartitionStrategy;

impl<T, A> Cluster<T, A> {
    /// Creates a new `Cluster` and returns the splits of items for its creating children.
    ///
    /// # Arguments
    ///
    /// - `depth` - The depth of the cluster in the tree.
    /// - `center_index` - The index of the center item in the full list of `items` in the tree instead of the local slice.
    /// - `items` - The local slice of items belonging to the cluster.
    /// - `metric` - The distance function to use.
    /// - `strategy` - The partitioning strategy to use.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    ///   - The new `Cluster`.
    ///   - A vector of tuples, one for each child cluster, each tuple containing:
    ///     - The center index of the child cluster in the full list of `items` in the tree.
    ///     - The slice of items belonging to the child cluster.
    ///
    /// # WARNING
    ///
    /// This function assumes that `items` is non-empty. In our implementation, this is checked *once* when creating the `Tree`.
    pub(crate) fn new<'a, Id, I, M, P>(
        depth: usize,
        center_index: usize,
        items: &'a mut [(Id, I)],
        metric: &M,
        strategy: &PartitionStrategy<P>,
    ) -> (Self, Vec<(usize, &'a mut [(Id, I)])>)
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
        P: Fn(&Self) -> bool,
    {
        ftlog::debug!(
            "Creating a new cluster at depth {depth} with center {center_index} and cardinality {}",
            items.len()
        );

        let (mut cluster, radius_index) = Self::new_leaf(depth, center_index, items, metric);
        if !strategy.should_partition(&cluster) {
            ftlog::debug!("Not partitioning the cluster at depth {}", cluster.depth);
            return (cluster, Vec::new());
        }
        ftlog::debug!("Partitioning the cluster at depth {}", cluster.depth);

        let (span, mut splits) = strategy.split(&mut items[1..], metric, radius_index, center_index);
        splits.sort_by_key(|&(c_index, _)| c_index);

        let child_cardinalities = splits.iter().map(|(_, c_items)| c_items.len()).collect::<Vec<_>>();
        ftlog::info!(
            "At depth {}, will create {} child clusters with {:?} cardinalities",
            depth,
            splits.len(),
            child_cardinalities
        );

        let child_center_indices = splits.iter().map(|&(c_index, _)| c_index).collect::<Vec<_>>();
        cluster.children = Some((child_center_indices.into_boxed_slice(), span));

        (cluster, splits)
    }

    /// Creates a new `Cluster` as a leaf.
    ///
    /// - If the number of `items` is 1, that item is the center, the radius is 0, and the LFD is 1.
    /// - If the number of `items` is 2, the 0th item is the center, the radius is the distance between the two items, and the LFD is 1.
    /// - If the number of `items` is greater than 2, this function will find the geometric median of the items (using an approximate method for large number of
    ///   items) and use it as the center of the cluster. It will swap the center item to the 0th index in the `items` slice. It will then compute the radius of
    ///   the cluster as the maximum distance from the center to any other item, and compute the LFD of the cluster.
    ///
    /// # Arguments
    ///
    /// - `depth` - The depth of the cluster in the tree.
    /// - `center_index` - The index of the center item in the full list of `items` in the tree instead of the local slice.
    /// - `items` - The local slice of items belonging to the cluster.
    /// - `metric` - The distance function to use.
    ///
    /// # Returns
    ///
    /// - A tuple containing the new `Cluster` and the index of the item that defines the radius of the cluster.
    ///
    /// # Side Effects
    ///
    /// This function will ensure that the center item, i.e. the geometric median, is at the 0th index in the `items` slice, modifying the slice in place if
    /// necessary.
    ///
    /// # WARNING
    ///
    /// This function:
    ///
    /// - assumes that `items` is non-empty. This is checked *once* when creating the `Tree` and ensured by the logic of the partitioning algorithms.
    /// - uses `unsafe` code to initialize the `annotation` to a dummy value. This is safe because this function is private and the annotation is later set in
    ///   `Tree::new` before any `Cluster` is used.
    /// - Sets the `parent_center_index` to the same value as `center_index`. This is a placeholder and is updated later in `Tree::new` before any `Cluster` is
    ///   used.
    fn new_leaf<Id, I, M>(depth: usize, center_index: usize, items: &mut [(Id, I)], metric: &M) -> (Self, usize)
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
    {
        ftlog::debug!(
            "Creating a new leaf cluster at depth {depth} with center {center_index} and cardinality {}",
            items.len()
        );

        if items.len() == 1 {
            let c = Self {
                depth,
                center_index,
                cardinality: 1,
                radius: T::zero(),
                lfd: 1.0, // By definition, a singleton has LFD of 1
                children: None,
                #[expect(unsafe_code)]
                // SAFETY: This is a private function and the annotation is later set in `Tree::new` before being used.
                annotation: unsafe { core::mem::zeroed() },
                parent_center_index: center_index,
            };
            return (c, 0);
        } else if items.len() == 2 {
            let radius = metric(&items[0].1, &items[1].1);
            let c = Self {
                depth,
                center_index,
                cardinality: 2,
                radius,
                lfd: 1.0, // By definition, a cluster with two items has LFD of 1
                children: None,
                #[expect(unsafe_code)]
                // SAFETY: This is a private function and the annotation is later set in `Tree::new` before being used.
                annotation: unsafe { core::mem::zeroed() },
                parent_center_index: center_index,
            };
            return (c, 1);
        }

        // Find and move the center (geometric median) to the front
        let n = num_items_for_geometric_median(items.len());
        swap_center_to_front(&mut items[..n], metric);

        let radial_distances = items.iter().skip(1).map(|(_, item)| metric(&items[0].1, item)).collect::<Vec<_>>();
        let (radius_index, radius) = radial_distances
            .iter()
            .enumerate()
            .max_by_key(|&(i, &d)| crate::utils::MaxItem(i, d))
            .map_or_else(|| unreachable!("items has enough elements"), |(i, &d)| (i, d));
        let lfd = lfd_estimate(&radial_distances, radius);

        let cluster = Self {
            depth,
            center_index,
            cardinality: items.len(),
            radius,
            lfd,
            children: None,
            #[expect(unsafe_code)]
            // SAFETY: This is a private function and the annotation is later set in `Tree::new` before being used.
            annotation: unsafe { core::mem::zeroed() },
            parent_center_index: center_index,
        };
        ftlog::debug!(
            "Created a new leaf cluster with depth {depth}, center {center_index}, cardinality {}, radius {radius}, and LFD {lfd}",
            cluster.cardinality
        );

        (cluster, radius_index)
    }
}

/// Computes the number of items to use for finding the geometric median.
#[expect(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub fn num_items_for_geometric_median(n_items: usize) -> usize {
    if n_items <= 100 {
        ftlog::debug!("Using all {n_items} items for finding the exact geometric median.");
        n_items
    } else {
        let n = if n_items <= 10_100 {
            let base = 100;
            let sqrt = ((n_items - 100) as f64).sqrt();
            base + sqrt as usize
        } else {
            let base = 200;
            let log = ((n_items - 10_100) as f64).log2();
            base + log as usize
        };
        ftlog::debug!("Using a random sample of size {n} out of {n_items} items for finding an approximate geometric median.");
        n
    }
}

/// Moves the center item (geometric median) to the 0th index in the slice.
pub fn swap_center_to_front<Id, I, T, M>(items: &mut [(Id, I)], metric: &M)
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    if items.len() > 2 {
        // Find the index of the item with the minimum total distance to all other items.
        let center_index = geometric_median(items, metric);
        items.swap(0, center_index);
    }
}
