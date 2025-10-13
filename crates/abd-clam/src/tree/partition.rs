//! Methods for recursively partitioning a `Node` to build a `Tree`.

use crate::DistanceValue;

use super::Node;

impl<T, A> Node<T, A> {
    /// Creates a new `Node` and recursively partitions it if it has more than two items.
    ///
    /// # WARNING
    ///
    /// This function assumes that the input `items` vector is non-empty. In our implementation, this is checked *once* when creating the `Tree`.
    pub(crate) fn new<Id, I, M>(depth: usize, mut items: Vec<(Id, I)>, metric: &M) -> (Self, Vec<(Id, I)>)
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
    {
        if items.len() == 1 {
            let node = Self {
                depth,
                center_index: 0,
                cardinality: 1,
                radius: T::zero(),
                lfd: 1.0, // By definition, a singleton has LFD of 1
                children: None,
                annotation: None,
            };
            return (node, items);
        } else if items.len() == 2 {
            let radius = metric(&items[0].1, &items[1].1);
            let node = Self {
                depth,
                center_index: 0,
                cardinality: 2,
                radius,
                lfd: 1.0, // By definition, a node with two items has LFD of 1
                children: None,
                annotation: None,
            };
            return (node, items);
        }

        let (center_id, center) = swap_remove_center(&mut items, metric);

        let radial_distances = items.iter().map(|(_, item)| metric(&center, item)).collect::<Vec<_>>();
        let (arg_radius, &radius) = radial_distances
            .iter()
            .enumerate()
            .max_by_key(|&(_, d)| crate::utils::MaxItem((), d))
            .unwrap_or_else(|| unreachable!("items must be non-empty"));
        let lfd = lfd_estimate(&radial_distances, radius);

        let (left_items, right_items, span) = bipolar_split(items, metric, Some(arg_radius));

        let (left_child, mut left_items) = Self::new(depth + 1, left_items, metric);
        let (mut right_child, mut right_items) = Self::new(depth + 1, right_items, metric);

        let mut items = Vec::with_capacity(left_items.len() + right_items.len() + 1);
        items.push((center_id, center));
        items.append(&mut left_items);
        items.append(&mut right_items);

        right_child.center_index += left_child.cardinality; // Adjust center index for right child after reassembling items

        let node = Self {
            depth,
            center_index: 0, // Center is now at index 0 after reassembling items
            cardinality: items.len(),
            radius,
            lfd,
            children: Some((Box::new([left_child, right_child]), span)),
            annotation: None,
        };

        (node, items)
    }
}

/// Moves the center item (geometric median) to the last index in the items array.
fn swap_remove_center<Id, I, T, M>(items: &mut Vec<(Id, I)>, metric: &M) -> (Id, I)
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    if items.len() < 3 {
        items.pop().unwrap_or_else(|| unreachable!("items must be non-empty"))
    } else {
        let arg_center = arg_gm(items, metric);
        items.swap_remove(arg_center)
    }
}

/// Returns the index of the geometric median of the given items.
///
/// The geometric median is the item that minimizes the sum of distances to
/// all other items in the slice.
///
/// The user must ensure that the items slice is not empty.
fn arg_gm<I, Id, T, M>(items: &[(Id, I)], metric: &M) -> usize
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
#[expect(clippy::needless_pass_by_value, clippy::type_complexity)]
fn bipolar_split<Id, I, T, M>(
    items: Vec<(Id, I)>,
    metric: &M,
    arg_left: Option<usize>,
) -> (Vec<(Id, I)>, Vec<(Id, I)>, T) {
    todo!()
}
