#![allow(dead_code)]

//! Utility functions for tests.

use core::cmp::Ordering;

use abd_clam::{Instance, VecDataset};
use distances::{
    number::{Float, UInt},
    Number,
};
use rand::prelude::*;

/// Euclidean distance between two vectors.
#[allow(clippy::ptr_arg)]
pub fn euclidean<T: Number, F: Float>(x: &Vec<T>, y: &Vec<T>) -> F {
    distances::vectors::euclidean(x, y)
}

/// Euclidean distance between two vectors.
#[allow(clippy::ptr_arg)]
pub fn euclidean_sq<T: Number>(x: &Vec<T>, y: &Vec<T>) -> T {
    distances::vectors::euclidean_sq(x, y)
}

/// Hamming distance between two Strings.
#[allow(clippy::ptr_arg)]
pub fn hamming<T: UInt>(x: &String, y: &String) -> T {
    distances::strings::hamming(x, y)
}

/// Levenshtein distance between two Strings.
#[allow(clippy::ptr_arg)]
pub fn levenshtein<T: UInt>(x: &String, y: &String) -> T {
    distances::strings::levenshtein(x, y)
}

/// Needleman-Wunsch distance between two Strings.
#[allow(clippy::ptr_arg)]
pub fn needleman_wunsch<T: UInt>(x: &String, y: &String) -> T {
    distances::strings::needleman_wunsch::nw_distance(x, y)
}

/// Generate a dataset with the given cardinality and dimensionality.
pub fn gen_dataset(
    cardinality: usize,
    dimensionality: usize,
    seed: u64,
    metric: fn(&Vec<f32>, &Vec<f32>) -> f32,
) -> VecDataset<Vec<f32>, f32, usize> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let data = symagen::random_data::random_tabular(cardinality, dimensionality, -1., 1., &mut rng);
    let name = "test".to_string();
    VecDataset::new(name, data, metric, false)
}

/// Generate a dataset from the given data.
pub fn gen_dataset_from<T: Number, U: Number, M: Instance>(
    data: Vec<Vec<T>>,
    metric: fn(&Vec<T>, &Vec<T>) -> U,
    metadata: Vec<M>,
) -> VecDataset<Vec<T>, U, M> {
    let name = "test".to_string();
    VecDataset::new(name, data, metric, false)
        .assign_metadata(metadata)
        .unwrap_or_else(|_| unreachable!())
}

/// Compute the recall of the nearest neighbors found.
///
/// Assumes that `linear_hits` is not empty.
pub fn compute_recall<T: Number>(mut hits: Vec<(usize, T)>, mut linear_hits: Vec<(usize, T)>) -> f32 {
    let num_hits = linear_hits.len();

    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    let mut hits = hits.into_iter().map(|(_, d)| d).peekable();

    linear_hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    let mut linear_hits = linear_hits.into_iter().map(|(_, d)| d).peekable();

    let mut num_common = 0;
    while let (Some(&hit), Some(&linear_hit)) = (hits.peek(), linear_hits.peek()) {
        if (hit - linear_hit).abs() < T::epsilon() {
            num_common += 1;
            hits.next();
            linear_hits.next();
        } else if hit < linear_hit {
            hits.next();
        } else {
            linear_hits.next();
        }
    }
    num_common.as_f32() / num_hits.as_f32()
}
