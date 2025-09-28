//! Utility functions for the crate.

use rayon::prelude::*;

use crate::DistanceValue;

mod ord_items;
mod sized_heap;

pub use ord_items::{MaxItem, MinItem};
pub use sized_heap::SizedHeap;

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
pub fn geometric_median<I, Id, T: DistanceValue, M: Fn(&I, &I) -> T>(items: &[(Id, I)], metric: &M) -> usize {
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
        .min_by_key(|&(i, v)| MinItem(i, v))
        .map_or_else(|| unreachable!("items must be non-empty"), |(i, _)| i)
}

/// Parallel version of [`geometric_median`](geometric_median).
pub fn par_geometric_median<
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
        .min_by_key(|&(i, v)| MinItem(i, v))
        .map_or_else(|| unreachable!("items must be non-empty"), |(i, _)| i)
}
