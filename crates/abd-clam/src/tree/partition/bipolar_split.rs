//! Bipolar split partitioning of items into two clusters.

use rayon::prelude::*;

use crate::DistanceValue;

/// A bipolar partition of items into two partitions.
#[derive(Debug)]
pub struct BipolarSplit<'a, Id, I, T> {
    /// The left partition of items. The 0th item is the left pole.
    pub l_items: &'a mut [(Id, I)],
    /// The right partition of items. The 0th item is the right pole.
    pub r_items: &'a mut [(Id, I)],
    /// The span of the partition (distance between the two poles).
    pub span: T,
    /// Distances from the left pole to the items in the left partition (excluding the left pole itself).
    pub l_distances: Vec<T>,
    /// Distances from the right pole to the items in the right partition (excluding the right pole itself).
    pub r_distances: Vec<T>,
}

/// The information we have about an initial pole for bipolar partitioning.
#[derive(Clone, Debug)]
pub enum InitialPole<T> {
    /// We have the index of the item farthest from the center to use as the left pole.
    RadialIndex(usize),
    /// The pole is the first item in the slice, and we have precomputed distances from it to all other items.
    Distances(Vec<T>),
}

impl<'a, Id, I, T> BipolarSplit<'a, Id, I, T>
where
    T: DistanceValue,
{
    /// Splits the given items into two partitions based on their distances to two poles.
    ///
    /// The two poles are chosen as follows:
    ///
    /// - If the `initial_pole` is `RadialIndex(i)`, the item at that index is chosen as the left pole.
    /// - If the `initial_pole` is `Distances(distances)`, the 0th item is chosen as the left pole, and the provided distances are used as distances from it to
    ///   all other items.
    /// - The right pole is then chosen as the item farthest from the left pole.
    ///
    /// The `span` of the partition is defined as the distance between the two poles.
    ///
    /// # Returns
    ///
    /// The `BipolarSplit` containing the two partitions of items, their distances to their respective poles, and the span of the partition.
    pub fn new<M>(items: &'a mut [(Id, I)], metric: &M, initial_pole: InitialPole<T>) -> Self
    where
        M: Fn(&I, &I) -> T,
    {
        if items.len() == 2 {
            ftlog::debug!("Splitting a cluster with only two items");
            // If there are only two items, just return them as the two partitions.
            let span = metric(&items[0].1, &items[1].1);
            let (l_items, r_items) = items.split_at_mut(1);
            let (l_distances, r_distances) = (vec![span], vec![span]);
            return Self {
                l_items,
                r_items,
                span,
                l_distances,
                r_distances,
            };
        }
        ftlog::debug!("Splitting a cluster with {} items", items.len());

        let mut left_distances = match initial_pole {
            InitialPole::RadialIndex(i) => {
                // Move the left pole to the 0th index in the slice
                items.swap(0, i);
                // Compute distances from the left pole to all other items
                items.iter().skip(1).map(|(_, item)| metric(&items[0].1, item)).collect::<Vec<_>>()
            }
            InitialPole::Distances(distances) => distances,
        };

        // Find the item farthest from the left pole
        let (right_pole_index, span) = left_distances
            .iter()
            .enumerate()
            .max_by_key(|&(i, &d)| crate::utils::MaxItem(i, d))
            .map_or_else(|| unreachable!("items has at least two elements"), |(i, &d)| (i + 1, d));

        // Move the right pole and its distance to the ends of their respective slices
        let last = items.len() - 1;
        items.swap(right_pole_index, last);
        left_distances.swap(right_pole_index - 1, last - 1);

        // Compute the distance from the right pole to all items
        let right_pole = &items[items.len() - 1].1;
        let mut left_right_distances = items
            .iter()
            .skip(1)
            .zip(left_distances)
            .take(items.len() - 2)
            .map(|((_, item), l)| (l, metric(right_pole, item)))
            .collect::<Vec<_>>();

        // Reorder the items in place by their distances to the two poles
        let mid = reorder_items_in_place(&mut items[1..last], &mut left_right_distances) + 1; // +1 to account for the left pole at index 0

        // Split the items and distances into the left and right partitions
        let (l_items, r_items) = items.split_at_mut(mid);
        let (l_distances, r_distances) = left_right_distances.split_at(mid - 1); // -1 to account for the left pole at index 0
        let l_distances = l_distances.iter().map(|&(l, _)| l).collect::<Vec<_>>();
        let r_distances = {
            // The first distance is just a placeholder for the right pole itself. We will swap it to the front to match with the right pole's position.
            let mut r_distances = core::iter::once(T::zero()).chain(r_distances.iter().map(|&(_, r)| r)).collect::<Vec<_>>();
            r_distances.swap_remove(0); // Remove the placeholder and move the first actual distance to the front
            r_distances
        };

        // Move the right pole to the 0th index of the right partition
        r_items.swap(0, r_items.len() - 1);

        Self {
            l_items,
            r_items,
            span,
            l_distances,
            r_distances,
        }
    }
}

impl<'a, Id, I, T> BipolarSplit<'a, Id, I, T>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
{
    /// Parallel version of [`Self::new`].
    pub fn par_new<M>(items: &'a mut [(Id, I)], metric: &M, initial_pole: InitialPole<T>) -> Self
    where
        M: Fn(&I, &I) -> T + Send + Sync,
    {
        if items.len() == 2 {
            ftlog::debug!("Splitting a cluster with only two items");
            // If there are only two items, just return them as the two partitions.
            let span = metric(&items[0].1, &items[1].1);
            let (l_items, r_items) = items.split_at_mut(1);
            let (l_distances, r_distances) = (vec![span], vec![span]);
            return Self {
                l_items,
                r_items,
                span,
                l_distances,
                r_distances,
            };
        }
        ftlog::debug!("Splitting a cluster with {} items", items.len());

        let mut left_distances = match initial_pole {
            InitialPole::RadialIndex(i) => {
                // Move the left pole to the 0th index in the slice
                items.swap(0, i);
                // Compute distances from the left pole to all other items
                items.par_iter().skip(1).map(|(_, item)| metric(&items[0].1, item)).collect::<Vec<_>>()
            }
            InitialPole::Distances(distances) => distances,
        };

        // Find the item farthest from the left pole
        let (right_pole_index, span) = left_distances
            .iter()
            .enumerate()
            .max_by_key(|&(i, &d)| crate::utils::MaxItem(i, d))
            .map_or_else(|| unreachable!("items has at least two elements"), |(i, &d)| (i + 1, d));

        // Move the right pole and its distance to the left pole to the end of their respective slices
        let last = items.len() - 1;
        items.swap(right_pole_index, last);
        left_distances.swap(right_pole_index - 1, last - 1);

        // Compute the distance from the right pole to all items
        let right_pole = &items[items.len() - 1].1;
        let mut left_right_distances = items
            .par_iter()
            .skip(1)
            .zip(left_distances)
            .take(items.len() - 2)
            .map(|((_, item), l)| (l, metric(right_pole, item)))
            .collect::<Vec<_>>();

        // Reorder the items in place by their distances to the two poles
        let mid = reorder_items_in_place(&mut items[1..last], &mut left_right_distances) + 1; // +1 to account for the left pole at index 0

        // Split the items and distances into the left and right partitions
        let (l_items, r_items) = items.split_at_mut(mid);
        let (l_distances, r_distances) = left_right_distances.split_at(mid - 1); // -1 to account for the left pole at index 0
        let l_distances = l_distances.iter().map(|&(l, _)| l).collect::<Vec<_>>();
        let r_distances = {
            // The first distance is just a placeholder for the right pole itself. We will swap it to the front to match with the right pole's position.
            let mut r_distances = core::iter::once(T::zero()).chain(r_distances.iter().map(|&(_, r)| r)).collect::<Vec<_>>();
            r_distances.swap_remove(0); // Remove the placeholder and move the first actual distance to the front
            r_distances
        };

        // Move the right pole to the 0th index of the right partition
        r_items.swap(0, r_items.len() - 1);

        Self {
            l_items,
            r_items,
            span,
            l_distances,
            r_distances,
        }
    }
}

/// Reorder the slice of items so that the items closer to left pole are on the left side of the slice and items closer to the right pole are on the right side,
/// returning `mid`, the index of the first item for the right pole.
///
/// # WARNING
///
/// This assumes that `items` and `distances` have the same length.
pub fn reorder_items_in_place<Id, I, T>(items: &mut [(Id, I)], distances: &mut [(T, T)]) -> usize
where
    T: DistanceValue,
{
    ftlog::debug!("Reordering {} items in place", items.len());

    let mut left = 0;
    let mut right = distances.len() - 1;

    // TODO(Najib): After testing, use unsafe code to remove bounds checks while indexing
    while left < right {
        // Increment `left` until we find an item for the right pole
        while left < distances.len() && distances[left].0 <= distances[left].1 {
            left += 1;
        }

        // Decrement `right` until we find an item for the left pole
        while right > 0 && distances[right].0 > distances[right].1 {
            right -= 1;
        }

        // If the two indices have crossed, we are done
        if left >= right {
            break;
        }

        // swap the items at the two indices
        items.swap(left, right);
        distances.swap(left, right);
        left += 1;
        right -= 1;
    }

    // TODO(Najib): Check if this last loop is even necessary

    // Increment `left` until we find the first item for the right pole
    while left < distances.len() && distances[left].0 <= distances[left].1 {
        left += 1;
    }

    left
}
