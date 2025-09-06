//! Tests for the `blas` accelerated distance functions.

#![allow(unused_imports)]

use float_eq::assert_float_eq;
use rand::prelude::*;
use test_case::test_case;

#[macro_use]
mod naive_impls;

/// Tests for BLAS accelerated Euclidean distance functions.
#[cfg(feature = "blas")]
#[test_case(10, 2; "10x2")]
#[test_case(100, 10; "100x10")]
#[test_case(100, 100; "100x100")]
#[test_case(100, 1000; "100x1000")]
fn blas_distances(car: usize, dim: usize) {
    let seed = 42;

    let data = naive_impls::gen_data::<f32>(car, dim, -1.0, 1.0, seed);
    for x in &data {
        for y in &data {
            assert_dist_eq!(x, y, l2_sq, distances_two::blas::euclidean_sq, 1e-5);
            assert_dist_eq!(x, y, l2, distances_two::blas::euclidean, 1e-5);
            assert_dist_eq!(x, y, cosine, distances_two::blas::cosine, 1e-5);
            assert_self_dist_eq!(x, norm_l2, distances_two::blas::norm_l2, 1e-5);
            assert_self_dist_eq!(y, norm_l2, distances_two::blas::norm_l2, 1e-5);
        }
    }

    let data = naive_impls::gen_data::<f64>(car, dim, -1.0, 1.0, seed);
    for x in &data {
        for y in &data {
            assert_dist_eq!(x, y, l2_sq, distances_two::blas::euclidean_sq, 1e-5);
            assert_dist_eq!(x, y, l2, distances_two::blas::euclidean, 1e-5);
            assert_dist_eq!(x, y, cosine, distances_two::blas::cosine, 1e-5);
            assert_self_dist_eq!(x, norm_l2, distances_two::blas::norm_l2, 1e-5);
            assert_self_dist_eq!(y, norm_l2, distances_two::blas::norm_l2, 1e-5);
        }
    }
}
