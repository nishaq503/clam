//! Methods for recursively partitioning a `Node` to build a `Tree`.

use crate::DistanceValue;

use super::Node;

impl<T, A> Node<T, A> {
    /// Creates a new `Node` and recursively partitions it if it has more than two items.
    ///
    /// # WARNING
    ///
    /// This function assumes that `items` is non-empty. In our implementation, this is checked *once* when creating the `Tree`.
    pub(crate) fn new_root<Id, I, M>(items: &mut [(Id, I)], metric: &M) -> Self
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
    {
        Self::new(0, 0, items, metric)
    }

    /// Creates a new `Node` and recursively partitions it if it has more than two items.
    ///
    /// # WARNING
    ///
    /// This function assumes that `items` is non-empty. In our implementation, this is checked *once* when creating the `Tree`.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    fn new<Id, I, M>(depth: usize, center_index: usize, items: &mut [(Id, I)], metric: &M) -> Self
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
    {
        if items.len() == 1 {
            return Self {
                depth,
                center_index,
                cardinality: 1,
                radius: T::zero(),
                lfd: 1.0, // By definition, a singleton has LFD of 1
                children: None,
                annotation: None,
            };
        } else if items.len() == 2 {
            let radius = metric(&items[0].1, &items[1].1);
            return Self {
                depth,
                center_index,
                cardinality: 2,
                radius,
                lfd: 1.0, // By definition, a node with two items has LFD of 1
                children: None,
                annotation: None,
            };
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

        let ([l_items, r_items], span) = bipolar_split(&mut items[1..], metric, Some(radius_index));

        let child_depth = depth + 1;
        let l_center_index = center_index + 1;
        let r_center_index = l_center_index + l_items.len();

        let l_child = Self::new(child_depth, l_center_index, l_items, metric);
        let r_child = Self::new(child_depth, r_center_index, r_items, metric);

        Self {
            depth,
            center_index,
            cardinality: items.len(),
            radius,
            lfd,
            children: Some((Box::new([l_child, r_child]), span)),
            annotation: None,
        }
    }
}

/// Moves the center item (geometric median) to the 0th index in the slice.
fn swap_center_to_front<Id, I, T, M>(items: &mut [(Id, I)], metric: &M)
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
fn gm_index<I, Id, T, M>(items: &[(Id, I)], metric: &M) -> usize
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
fn bipolar_split<'a, Id, I, T, M>(
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

    // Move the right pole and its distance to the left pole to the end of their respective slices
    let last = items.len() - 1;
    items.swap(right_pole_index, last);
    left_distances.swap(right_pole_index - 1, last - 1);

    // Compute the distance from the right pole to all items
    let right_pole = &items[items.len() - 1].1;
    let left_right_distances = items
        .iter()
        .skip(1)
        .zip(left_distances)
        .take(items.len() - 2)
        .map(|((_, item), l)| (l, metric(right_pole, item)))
        .collect::<Vec<_>>();

    // Reorder the items in place by their distances to the two poles
    let mid = reorder_items_in_place(&mut items[1..last], &left_right_distances) + 1; // +1 to account for the left pole at index 0

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
pub fn reorder_items_in_place<Id, I, T>(items: &mut [(Id, I)], distances: &[(T, T)]) -> usize
where
    T: DistanceValue,
{
    let mut left = 0;
    let mut right = items.len() - 1;

    // TODO(Najib): After tesing, use unsafe code to remove bounds checks while indexing

    loop {
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
        left += 1;
        right -= 1;
    }
    left
}
