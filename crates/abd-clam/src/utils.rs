//! Utility functions for the crate.

use core::cmp::Ordering;

use crate::{MaxItem, MinItem};

/// Find the median value using the quickselect algorithm.
///
/// If the number of elements is odd, the median is the middle element.
/// If the number of elements is even, the median should be the average of the
/// two middle elements, but this implementation returns the lower of the two
/// middle elements.
///
/// Source: <https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html>
///
/// # Arguments
///
/// * `data` - The data to find the median of.
#[expect(dead_code)]
pub fn median<T: PartialOrd + Copy>(data: &[T]) -> Option<T> {
    let size = data.len();

    match size {
        even if even % 2 == 0 => select(data, (even / 2) - 1),
        odd => select(data, odd / 2),
    }
}

/// A helper function for the median function below.
///
/// This function selects the kth smallest element from the given data.
///
/// # Arguments
///
/// * `data` - The data to select the kth smallest element from.
/// * `k` - The index of the element to select.
pub fn select<T: PartialOrd + Copy>(data: &[T], k: usize) -> Option<T> {
    let part = partition(data);

    match part {
        None => None,
        Some((left, pivot, right)) => {
            let pivot_idx = left.len();

            match pivot_idx.cmp(&k) {
                Ordering::Equal => Some(pivot),
                Ordering::Greater => select(&left, k),
                Ordering::Less => select(&right, k - (pivot_idx + 1)),
            }
        }
    }
}

/// A helper function for the median function below.
///
/// This function partitions the given data into three parts:
/// - A slice of all values less than the pivot value.
/// - The pivot value.
/// - A slice of all values greater than the pivot value.
///
/// # Arguments
///
/// * `data` - The data to partition.
pub fn partition<T: PartialOrd + Copy>(data: &[T]) -> Option<(Vec<T>, T, Vec<T>)> {
    data.split_first().map(|(&pivot, tail)| {
        let (left, right) = tail.iter().fold((vec![], vec![]), |(mut left, mut right), &next| {
            if next < pivot {
                left.push(next);
            } else {
                right.push(next);
            }
            (left, right)
        });

        (left, pivot, right)
    })
}

/// Use the stringzilla implementation of the Levenshtein distance.
///
/// # Panics
///
/// - If the device could not be created.
/// - If the Levenshtein distance engine could not be created.
/// - If the distance could not be computed.
#[cfg(feature = "musals")]
#[allow(clippy::unwrap_used)]
pub fn sz_lev_builder<I: AsRef<[u8]>>() -> impl Fn(&I, &I) -> usize {
    let device = stringzilla::szs::DeviceScope::default().unwrap();
    let szla_engine = stringzilla::szs::LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
    move |x, y| szla_engine.compute(&device, &[x], &[y]).unwrap()[0]
}
