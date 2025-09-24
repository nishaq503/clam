//! Utility functions fromo the core module.

mod ord_items;
mod sized_heap;

pub use ord_items::*;
pub use sized_heap::*;

/// Return the number of samples to take from the given population size so as to
/// achieve linear time complexity for geometric median estimation.
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

// /// Return the index of the minimum value in the given slice of values.
// ///
// /// Incomplete values are ordered as larger than all other values.
// pub fn arg_min<T: PartialOrd, I: IntoIterator<Item = T>>(values: I) -> Option<usize> {
//     values
//         .into_iter()
//         .enumerate()
//         .map(|(i, v)| MinItem(i, v))
//         .min()
//         .map(|MinItem(i, _)| i)
// }

// /// Return the index of the maximum value in the given slice of values.
// ///
// /// Incomplete values are ordered as smaller than all other values.
// pub fn arg_max<T: PartialOrd, I: IntoIterator<Item = T>>(values: I) -> Option<usize> {
//     values
//         .into_iter()
//         .enumerate()
//         .map(|(i, v)| MaxItem(i, v))
//         .max()
//         .map(|MaxItem(i, _)| i)
// }
