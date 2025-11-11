//! Methods for recursively partitioning a `Cluster` to build a `Tree`.

use crate::{DistanceValue, PartitionStrategy, utils::SizedHeap};

use super::Cluster;

impl<T, A> Cluster<T, A> {
    /// Creates a new `Cluster` and recursively partitions it if it has more than two items.
    ///
    /// # WARNING
    ///
    /// This function assumes that `items` is non-empty. In our implementation, this is checked *once* when creating the `Tree`.
    pub(crate) fn new_root<Id, I, M, P, Ann>(
        items: &mut [(Id, I)],
        metric: &M,
        strategy: &PartitionStrategy<P>,
        annotator: &Ann,
    ) -> Self
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
        P: Fn(&Self) -> bool,
        Ann: Fn(&Self) -> Option<A>,
    {
        Self::new(0, 0, items, metric, strategy, annotator)
    }

    /// Creates a new `Cluster` and recursively partitions it if it has more than two items.
    ///
    /// # WARNING
    ///
    /// This function assumes that `items` is non-empty. In our implementation, this is checked *once* when creating the `Tree`.
    fn new<Id, I, M, P, Ann>(
        depth: usize,
        center_index: usize,
        items: &mut [(Id, I)],
        metric: &M,
        strategy: &PartitionStrategy<P>,
        annotator: &Ann,
    ) -> Self
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
        P: Fn(&Self) -> bool,
        Ann: Fn(&Self) -> Option<A>,
    {
        let (mut cluster, radius_index) = Self::new_leaf(depth, center_index, items, metric);
        if !strategy.should_partition(&cluster) {
            return cluster;
        }

        let ([l_items, r_items], span) = bipolar_split(&mut items[1..], metric, Some(radius_index));
        let (lci, rci) = (center_index + 1, center_index + 1 + l_items.len());

        let mut child_items = if let Some(n_children) = strategy.branching_factor.for_cardinality(cluster.cardinality) {
            let mut child_items = SizedHeap::new(Some(n_children));
            let nl = l_items.len();
            child_items.push((l_items, (nl, lci)));
            let nr = r_items.len();
            child_items.push((r_items, (nr, rci)));

            while !child_items.is_full() {
                let (items, (_, ci)) = child_items
                    .pop()
                    .unwrap_or_else(|| unreachable!("child_items is not empty"));
                if items.len() < 2 {
                    break;
                }
                let ([l_items, r_items], _) = bipolar_split(items, metric, None);

                let nl = l_items.len();
                let nr = r_items.len();
                let lci = ci;
                let rci = ci + nl;

                child_items.push((l_items, (nl, lci)));
                child_items.push((r_items, (nr, rci)));
            }

            child_items
                .take_items()
                .map(|(c_items, (_, ci))| (ci, c_items))
                .collect::<Vec<_>>()
        } else {
            let max_span = strategy.span_reduction.max_child_span_for(span);

            let mut child_items = SizedHeap::new(None);
            let l_span = span_estimate(l_items, metric);
            child_items.push((l_items, (l_span, lci)));
            let r_span = span_estimate(r_items, metric);
            child_items.push((r_items, (r_span, rci)));

            while child_items.peek().is_some_and(|(_, (s, _))| *s > max_span) {
                let (items, (_, ci)) = child_items
                    .pop()
                    .unwrap_or_else(|| unreachable!("child_items is not empty"));
                if items.len() < 2 {
                    break;
                }

                let ([l_items, r_items], _) = bipolar_split(items, metric, None);

                let l_span = span_estimate(l_items, metric);
                let r_span = span_estimate(r_items, metric);
                let lci = ci;
                let rci = ci + l_items.len();

                child_items.push((l_items, (l_span, lci)));
                child_items.push((r_items, (r_span, rci)));
            }

            child_items
                .take_items()
                .map(|(c_items, (_, ci))| (ci, c_items))
                .collect::<Vec<_>>()
        };
        child_items.sort_by_key(|&(c_index, _)| c_index);

        let children = child_items
            .into_iter()
            .map(|(c_index, c_items)| Self::new(depth + 1, c_index, c_items, metric, strategy, annotator))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        cluster.children = Some((children, span));

        cluster.annotation = annotator(&cluster);

        cluster
    }

    /// Creates a new `Cluster` as a leaf.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    fn new_leaf<Id, I, M>(depth: usize, center_index: usize, items: &mut [(Id, I)], metric: &M) -> (Self, usize)
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
    {
        if items.len() == 1 {
            let c = Self {
                depth,
                center_index,
                cardinality: 1,
                radius: T::zero(),
                lfd: 1.0, // By definition, a singleton has LFD of 1
                children: None,
                annotation: None,
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
                annotation: None,
            };
            return (c, 1);
        }

        if items.len() <= 100 {
            // For small number of items, find the exact geometric median
            swap_center_to_front(items, metric);
        } else {
            let n = 100 + ((items.len() - 100) as f64).sqrt() as usize;
            // For large number of items, find an approximate geometric median using a random sample of size n
            swap_center_to_front(&mut items[..n], metric);
        }

        let radial_distances = items
            .iter()
            .skip(1)
            .map(|(_, item)| metric(&items[0].1, item))
            .collect::<Vec<_>>();
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
            annotation: None,
        };

        (cluster, radius_index)
    }
}

/// Moves the center item (geometric median) to the 0th index in the slice.
pub fn swap_center_to_front<Id, I, T, M>(items: &mut [(Id, I)], metric: &M)
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    if items.len() > 2 {
        let center_index = gm_index(items, metric);
        items.swap(0, center_index);
    }
}

/// Returns the index of the geometric median of the given items.
///
/// The geometric median is the item that minimizes the sum of distances to
/// all other items in the slice.
///
/// The user must ensure that the items slice is not empty.
pub fn gm_index<I, Id, T, M>(items: &[(Id, I)], metric: &M) -> usize
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
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
        .min_by_key(|&(i, v)| crate::utils::MinItem(i, v))
        .map_or_else(|| unreachable!("items must be non-empty"), |(i, _)| i)
}

/// Estimates the Local Fractal Dimension (LFD) using the distances of items from a center, and the maximum value among those distances.
///
/// This uses the formula `log2(N / n)`, where `N` is `distances.len() + 1` (the total number of items including the center), and `n` is the number of distances
/// that are less than or equal to `radius / 2` plus one (to account for the center).
///
/// If the radius is zero or if there are no items within half the radius, the LFD is, by definition, `1.0`.
#[expect(clippy::cast_precision_loss)]
pub fn lfd_estimate<T>(distances: &[T], radius: T) -> f64
where
    T: DistanceValue,
{
    let half_radius = radius.half();
    if distances.len() < 2 || half_radius.is_zero() {
        // In all three of the following cases, we define LFD to be 1.0:
        //   - No non-center items (singleton cluster)
        //   - One non-center item (cluster with two items)
        //   - Radius is zero or too small to be represented as a non-zero value
        1.0
    } else {
        // The cluster has at least 2 non-center items, so LFD computation is meaningful.
        let half_count = distances.iter().filter(|&&d| d <= half_radius).count();
        ((distances.len() + 1) as f64 / ((half_count + 1) as f64)).log2()
    }
}

/// Splits the given items into two partitions based on their distances to two poles.
///
/// The two poles are chosen as follows:
///
/// - If `arg_left` is provided, the item at that index is chosen as the left pole.
/// - If `arg_left` is `None`, an arbitrary item (the first one) is temporarily chosen, and the item farthest from it is chosen as the left pole.
/// - The right pole is then chosen as the item farthest from the left pole.
///
/// The `span` of the partition is defined as the distance between the two poles.
///
/// The items are then partitioned based on their distances to the two poles with ties going to the left partition.
/// Finally, the poles are added back into their respective partitions, as the last item in each.
///
/// # Returns
///
/// - An array containing the two partitions of items.
/// - The span of the partition (distance between the two poles).
#[expect(clippy::tuple_array_conversions)]
pub fn bipolar_split<'a, Id, I, T, M>(
    items: &'a mut [(Id, I)],
    metric: &M,
    left_pole_index: Option<usize>,
) -> ([&'a mut [(Id, I)]; 2], T)
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    if items.len() == 2 {
        // If there are only two items, just return them as the two partitions.
        let span = metric(&items[0].1, &items[1].1);
        let (left, right) = items.split_at_mut(1);
        return ([left, right], span);
    }

    let left_pole_index = left_pole_index.unwrap_or_else(|| {
        // Find the item farthest from the first item.
        items
            .iter()
            .enumerate()
            .skip(1)
            .max_by_key(|&(_, (_, item))| crate::utils::MaxItem((), metric(&items[0].1, item)))
            .map_or_else(|| unreachable!("items must be non-empty"), |(i, _)| i)
    });

    // Move the left pole to the 0th index in the slice
    items.swap(0, left_pole_index);

    // Compute distances from the left pole to all other items
    let left_pole = &items[0].1;
    let mut left_distances = items
        .iter()
        .skip(1)
        .map(|(_, item)| metric(left_pole, item))
        .collect::<Vec<_>>();

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

    // split the items slice into the left and right partitions
    let (left, right) = items.split_at_mut(mid);

    // Move the right pole to the 0th index in the right slice
    right.swap(0, right.len() - 1);

    ([left, right], span)
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
    assert_eq!(
        items.len(),
        distances.len(),
        "items and distances must have the same length"
    );

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

/// Estimates the Span (maximum distance between any two items) of the given items using a heuristic approach.
pub fn span_estimate<Id, I, T, M>(items: &[(Id, I)], metric: &M) -> T
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    match items.len().cmp(&2) {
        core::cmp::Ordering::Less => T::zero(),
        core::cmp::Ordering::Equal => metric(&items[0].1, &items[1].1),
        core::cmp::Ordering::Greater => {
            let temp_pole_index = 0;
            let left_pole_index = items
                .iter()
                .enumerate()
                .skip(1)
                .max_by_key(|&(_, (_, item))| crate::utils::MaxItem((), metric(&items[temp_pole_index].1, item)))
                .map_or_else(|| unreachable!("items must be non-empty"), |(i, _)| i);
            items
                .iter()
                .enumerate()
                .map(|(i, (_, item))| (i, metric(&items[left_pole_index].1, item)))
                .max_by_key(|&(i, d)| crate::utils::MaxItem(i, d))
                .map_or_else(|| unreachable!("items has at least two elements"), |(_, d)| d)
        }
    }
}
