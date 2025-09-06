//! Tests for the `simd` accelerated distance functions.

#![allow(unused_imports)]

use float_eq::assert_float_eq;
use test_case::test_case;

#[macro_use]
mod naive_impls;

/// Tests for SIMD accelerated Euclidean distance functions.
#[cfg(any(feature = "simd-128", feature = "simd-256", feature = "simd-512", feature = "simd-1024"))]
#[test_case(10, 2; "10x2")]
#[test_case(100, 10; "100x10")]
#[test_case(100, 100; "100x100")]
#[test_case(100, 1000; "100x1000")]
fn simd_distances(car: usize, dim: usize) {
    let seed = 42;

    let data = naive_impls::gen_data::<f32>(car, dim, -1.0, 1.0, seed);
    let tol = 1e-5;
    for x in &data {
        for y in &data {
            #[cfg(feature = "simd-128")]
            check_approx_eq_f32(x, y, tol);

            #[cfg(feature = "simd-256")]
            check_approx_eq_f32(x, y, tol);

            #[cfg(feature = "simd-512")]
            check_approx_eq_f32(x, y, tol);

            #[cfg(feature = "simd-1024")]
            check_approx_eq_f32(x, y, tol);
        }
    }

    let data = naive_impls::gen_data::<f64>(car, dim, -1.0, 1.0, seed);
    let tol = 1e-5;
    for x in &data {
        for y in &data {
            #[cfg(feature = "simd-128")]
            check_approx_eq_f64(x, y, tol);

            #[cfg(feature = "simd-256")]
            check_approx_eq_f64(x, y, tol);

            #[cfg(feature = "simd-512")]
            check_approx_eq_f64(x, y, tol);

            #[cfg(feature = "simd-1024")]
            check_approx_eq_f64(x, y, tol);
        }
    }
}

#[cfg(any(feature = "simd-128", feature = "simd-256", feature = "simd-512", feature = "simd-1024"))]
fn check_approx_eq_f32(x: &[f32], y: &[f32], tol: f32) {
    assert_dist_eq!(x, y, l2_sq, distances_two::simd::euclidean_sq, tol);
    assert_dist_eq!(x, y, l2, distances_two::simd::euclidean, tol);
    assert_dist_eq!(x, y, cosine, distances_two::simd::cosine, tol);
    assert_self_dist_eq!(x, norm_l2, distances_two::simd::norm_l2, tol);
    assert_self_dist_eq!(y, norm_l2, distances_two::simd::norm_l2, tol);

    assert_simd_dist_eq!(x, y, l2_sq, euclidean_sq, tol, f32);
    assert_simd_dist_eq!(x, y, l2, euclidean, tol, f32);
    assert_simd_dist_eq!(x, y, cosine, cosine, tol, f32);
    assert_simd_self_dist_eq!(x, norm_l2, norm_l2, tol, f32);
    assert_simd_self_dist_eq!(y, norm_l2, norm_l2, tol, f32);
}

#[cfg(any(feature = "simd-128", feature = "simd-256", feature = "simd-512", feature = "simd-1024"))]
fn check_approx_eq_f64(x: &[f64], y: &[f64], tol: f64) {
    assert_dist_eq!(x, y, l2_sq, distances_two::simd::euclidean_sq, tol);
    assert_dist_eq!(x, y, l2, distances_two::simd::euclidean, tol);
    assert_dist_eq!(x, y, cosine, distances_two::simd::cosine, tol);
    assert_self_dist_eq!(x, norm_l2, distances_two::simd::norm_l2, tol);
    assert_self_dist_eq!(y, norm_l2, distances_two::simd::norm_l2, tol);

    assert_simd_dist_eq!(x, y, l2_sq, euclidean_sq, tol, f64);
    assert_simd_dist_eq!(x, y, l2, euclidean, tol, f64);
    assert_simd_dist_eq!(x, y, cosine, cosine, tol, f64);
    assert_simd_self_dist_eq!(x, norm_l2, norm_l2, tol, f64);
    assert_simd_self_dist_eq!(y, norm_l2, norm_l2, tol, f64);
}
