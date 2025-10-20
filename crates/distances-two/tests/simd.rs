//! Tests for the `simd` accelerated distance functions.

use float_eq::assert_float_eq;
use rand::prelude::*;
use test_case::test_case;

mod naive_impls;

/// Tests for SIMD accelerated Euclidean distance functions.
#[test_case(10, 2; "10x2")]
#[test_case(100, 10; "100x10")]
#[test_case(100, 100; "100x100")]
#[test_case(100, 1000; "100x1000")]
fn simd_f32(car: usize, dim: usize) {
    let seed = 42;
    let (min_val, max_val) = (-1_f32, 1_f32);

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let data: Vec<Vec<f32>> = (0..car)
        .map(|_| {
            (0..dim)
                .map(|_| rand::Rng::random_range(&mut rng, min_val..=max_val))
                .collect()
        })
        .collect();

    for x in &data {
        for y in &data {
            #[cfg(feature = "simd-128")]
            check_approx_eq(x, y, 1e-5);

            #[cfg(feature = "simd-256")]
            check_approx_eq(x, y, 1e-5);

            #[cfg(feature = "simd-512")]
            check_approx_eq(x, y, 1e-5);
        }
    }
}

fn check_approx_eq(x: &[f32], y: &[f32], tol: f32) {
    let e_l2_sq = naive_impls::l2_sq(x, y);

    let a_l2_sq: f32 = distances_two::simd::euclidean_sq(x, y);
    let ratio = if e_l2_sq != 0.0 {
        (e_l2_sq - a_l2_sq).abs() / e_l2_sq.abs()
    } else {
        0.0
    };
    assert_float_eq!(ratio, 0.0, abs <= tol);

    let e_l2 = naive_impls::l2(x, y);
    let a_l2: f32 = distances_two::simd::euclidean(x, y);
    let ratio = if e_l2 != 0.0 {
        (e_l2 - a_l2).abs() / e_l2.abs()
    } else {
        0.0
    };
    assert_float_eq!(ratio, 0.0, abs <= tol);
}
