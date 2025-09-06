//! Methods for recursively partitioning a `Cluster` to build a `Tree`.

use rayon::prelude::*;

use crate::{
    Cluster, DistanceValue,
    utils::{lfd_estimate, par_geometric_median},
};

use super::{num_items_for_geometric_median, strategy::PartitionStrategy};

impl<T, A> Cluster<T, A> {
    /// Parallel version of [`Self::new`].
    pub(crate) fn par_new<'a, Id, I, M, P>(
        depth: usize,
        center_index: usize,
        items: &'a mut [(Id, I)],
        metric: &M,
        strategy: &PartitionStrategy<P>,
    ) -> (Self, Vec<(usize, &'a mut [(Id, I)])>)
    where
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
        P: Fn(&Self) -> bool + Send + Sync,
    {
        ftlog::debug!(
            "Creating a new cluster at depth {depth} with center {center_index} and cardinality {}",
            items.len()
        );

        let (mut cluster, radius_index) = Self::par_new_leaf(depth, center_index, items, metric);
        if !strategy.should_partition(&cluster) {
            ftlog::debug!("Not partitioning the cluster at depth {}", cluster.depth);
            return (cluster, Vec::new());
        }
        ftlog::debug!("Partitioning the cluster at depth {}", cluster.depth);

        let (span, mut splits) = strategy.par_split(&mut items[1..], metric, radius_index, center_index);
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
    fn par_new_leaf<Id, I, M>(depth: usize, center_index: usize, items: &mut [(Id, I)], metric: &M) -> (Self, usize)
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
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
                // SAFETY: This is a private function and the annotation is later set in `par_new` before being used.
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
                // SAFETY: This is a private function and the annotation is later set in `par_new` before being used.
                annotation: unsafe { core::mem::zeroed() },
                parent_center_index: center_index,
            };
            return (c, 1);
        }

        // Find and move the center (geometric median) to the front
        let n = num_items_for_geometric_median(items.len());
        par_swap_center_to_front(&mut items[..n], metric);

        let radial_distances = items.par_iter().skip(1).map(|(_, item)| metric(&items[0].1, item)).collect::<Vec<_>>();
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
            // SAFETY: This is a private function and the annotation is later set in `par_new` before being used.
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

/// Moves the center item (geometric median) to the 0th index in the slice.
pub fn par_swap_center_to_front<Id, I, T, M>(items: &mut [(Id, I)], metric: &M)
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    if items.len() > 2 {
        // Find the index of the item with the minimum total distance to all other items.
        let center_index = par_geometric_median(items, metric);
        items.swap(0, center_index);
    }
}
