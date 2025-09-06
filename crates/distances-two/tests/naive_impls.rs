//! Naive implementations of various distance metrics for testing purposes.

#![allow(dead_code, unused_macros)]

use rand::prelude::*;

/// A macro that expands to distance equality assertions.
macro_rules! assert_dist_eq {
    ($x:expr, $y:expr, $naive_fn:ident, $dist_fn:expr, $tol:expr) => {{
        let e_dist = naive_impls::$naive_fn($x, $y);
        let a_dist = $dist_fn($x, $y);
        let ratio = if e_dist == 0.0 { 0.0 } else { (e_dist - a_dist).abs() / e_dist.abs() };
        assert_float_eq!(ratio, 0.0, abs <= $tol);
    }};
}

/// A macro that expands to distance equality assertions with SIMD lanes.
macro_rules! assert_dist_lanes_eq {
    ($x:expr, $y:expr, $naive_fn:ident, $dist_fn:ident, $tol:expr, $ty:ty, $lanes:expr) => {{
        let e_dist = naive_impls::$naive_fn::<$ty>($x, $y);
        let a_dist = distances_two::std_simd::$dist_fn::<$ty, _, $lanes>($x, $y);
        let ratio = if e_dist == 0.0 { 0.0 } else { (e_dist - a_dist).abs() / e_dist.abs() };
        assert_float_eq!(ratio, 0.0, abs <= $tol);
    }};
}

/// A macro that expands to distance equality assertions with SIMD lanes.
macro_rules! assert_self_dist_lanes_eq {
    ($x:expr, $naive_fn:ident, $dist_fn:ident, $tol:expr, $ty:ty, $lanes:expr) => {{
        let e_dist = naive_impls::$naive_fn::<$ty>($x);
        let a_dist = distances_two::std_simd::$dist_fn::<$ty, _, $lanes>($x);
        let ratio = if e_dist == 0.0 { 0.0 } else { (e_dist - a_dist).abs() / e_dist.abs() };
        assert_float_eq!(ratio, 0.0, abs <= $tol);
    }};
}

/// A macro that expands to distance equality assertions with SIMD lanes.
macro_rules! assert_simd_dist_eq {
    ($x:expr, $y:expr, $naive_fn:ident, $dist_fn:ident, $tol:expr, $ty:ty) => {{
        assert_dist_lanes_eq!($x, $y, $naive_fn, $dist_fn, $tol, $ty, 2);
        assert_dist_lanes_eq!($x, $y, $naive_fn, $dist_fn, $tol, $ty, 4);
        assert_dist_lanes_eq!($x, $y, $naive_fn, $dist_fn, $tol, $ty, 8);
        assert_dist_lanes_eq!($x, $y, $naive_fn, $dist_fn, $tol, $ty, 16);
        assert_dist_lanes_eq!($x, $y, $naive_fn, $dist_fn, $tol, $ty, 32);
        assert_dist_lanes_eq!($x, $y, $naive_fn, $dist_fn, $tol, $ty, 64);
    }};
}

/// A macro that expands to distance equality assertions with SIMD lanes.
macro_rules! assert_simd_self_dist_eq {
    ($x:expr, $naive_fn:ident, $dist_fn:ident, $tol:expr, $ty:ty) => {{
        assert_self_dist_lanes_eq!($x, $naive_fn, $dist_fn, $tol, $ty, 2);
        assert_self_dist_lanes_eq!($x, $naive_fn, $dist_fn, $tol, $ty, 4);
        assert_self_dist_lanes_eq!($x, $naive_fn, $dist_fn, $tol, $ty, 8);
        assert_self_dist_lanes_eq!($x, $naive_fn, $dist_fn, $tol, $ty, 16);
        assert_self_dist_lanes_eq!($x, $naive_fn, $dist_fn, $tol, $ty, 32);
        assert_self_dist_lanes_eq!($x, $naive_fn, $dist_fn, $tol, $ty, 64);
    }};
}

/// A macro that expands to distance equality assertions.
macro_rules! assert_self_dist_eq {
    ($x:expr, $naive_fn:ident, $dist_fn:expr, $tol:expr) => {{
        let e_dist = naive_impls::$naive_fn($x);
        let a_dist = $dist_fn($x);
        let ratio = if e_dist == 0.0 { 0.0 } else { (e_dist - a_dist).abs() / e_dist.abs() };
        assert_float_eq!(ratio, 0.0, abs <= $tol);
    }};
}

/// Asserts that several distance functions produce approximately equal results.
macro_rules! check_distances {
    ($x:expr, $y:expr, $tol:expr) => {
        assert_dist_eq!($x, $y, l1, distances_two::vectors::manhattan, $tol);
        assert_dist_eq!($x, $y, l2, distances_two::vectors::euclidean, $tol);
        assert_dist_eq!($x, $y, l2_sq, distances_two::vectors::euclidean_sq, $tol);
        assert_dist_eq!($x, $y, l_inf, distances_two::vectors::chebyshev, $tol);

        let mink_l1 = |a, b| distances_two::vectors::minkowski(a, b, 1.0);
        assert_dist_eq!($x, $y, l1, mink_l1, $tol);
        let mink_l2 = |a, b| distances_two::vectors::minkowski(a, b, 2.0);
        assert_dist_eq!($x, $y, l2, mink_l2, $tol);
        let mink_l3 = |a, b| distances_two::vectors::minkowski(a, b, 3.0);
        assert_dist_eq!($x, $y, l3, mink_l3, $tol);
        let mink_l4 = |a, b| distances_two::vectors::minkowski(a, b, 4.0);
        assert_dist_eq!($x, $y, l4, mink_l4, $tol);

        assert_dist_eq!($x, $y, dot, distances_two::vectors::dot_product, $tol);
        assert_dist_eq!($x, $y, cosine, distances_two::vectors::cosine, $tol);

        let e_norm_l2_x = naive_impls::norm_l2($x);
        let a_norm_l2_x = distances_two::vectors::norm_l2($x);
        let delta_x = (e_norm_l2_x - a_norm_l2_x).abs();
        assert_float_eq!(delta_x, 0.0, abs <= $tol);

        let e_norm_l2_y = naive_impls::norm_l2($y);
        let a_norm_l2_y = distances_two::vectors::norm_l2($y);
        let delta_y = (e_norm_l2_y - a_norm_l2_y).abs();
        assert_float_eq!(delta_y, 0.0, abs <= $tol);
    };
}

/// Returns an iterator over the absolute differences between corresponding elements of two vectors.
pub fn iter_abs_diff<'a, F: num_traits::Float>(x: &'a [F], y: &'a [F]) -> impl Iterator<Item = F> + 'a {
    x.iter().zip(y.iter()).map(|(&x, &y)| x - y).map(F::abs)
}

/// Computes the Manhattan (L1) distance between two vectors.
pub fn l1<F: num_traits::Float>(x: &[F], y: &[F]) -> F {
    iter_abs_diff(x, y).fold(F::zero(), |acc, d| acc + d)
}

/// Computes the squared Euclidean (L2) distance between two vectors.
pub fn l2_sq<F: num_traits::Float>(x: &[F], y: &[F]) -> F {
    iter_abs_diff(x, y).map(|d| d * d).fold(F::zero(), |acc, d2| acc + d2)
}

/// Computes the Euclidean (L2) distance between two vectors.
pub fn l2<F: num_traits::Float>(x: &[F], y: &[F]) -> F {
    l2_sq(x, y).sqrt()
}

/// Computes the L3-norm between two vectors.
pub fn l3<F: num_traits::Float>(x: &[F], y: &[F]) -> F {
    iter_abs_diff(x, y)
        .map(|d| d.powi(3))
        .fold(F::zero(), |acc, d3| acc + d3)
        .powf((F::one() + F::one() + F::one()).recip())
}

/// Computes the L4-norm between two vectors.
pub fn l4<F: num_traits::Float>(x: &[F], y: &[F]) -> F {
    iter_abs_diff(x, y).map(|d| d.powi(4)).fold(F::zero(), |acc, d4| acc + d4).sqrt().sqrt()
}

/// Computes the Chebyshev (L∞) distance between two vectors.
pub fn l_inf<F: num_traits::Float>(x: &[F], y: &[F]) -> F {
    iter_abs_diff(x, y).fold(F::zero(), |acc, d| if d > acc { d } else { acc })
}

/// Computes the dot product of two vectors.
pub fn dot<F: num_traits::Float>(x: &[F], y: &[F]) -> F {
    x.iter().zip(y.iter()).map(|(&x, &y)| x * y).fold(F::zero(), |acc, p| acc + p)
}

/// Computes the l2-norm of a vector.
pub fn norm_l2<F: num_traits::Float>(x: &[F]) -> F {
    dot(x, x).sqrt()
}

/// Computes the cosine distance between two vectors.
pub fn cosine<F: num_traits::Float>(x: &[F], y: &[F]) -> F {
    let xy = dot(x, y);
    let xx = dot(x, x);
    let yy = dot(y, y);
    F::one() - xy / (xx * yy).sqrt()
}

/// Generates random data for testing.
pub fn gen_data<F: num_traits::Float + rand::distr::uniform::SampleUniform>(car: usize, dim: usize, min_val: F, max_val: F, seed: u64) -> Vec<Vec<F>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..car)
        .map(|_| (0..dim).map(|_| rand::Rng::random_range(&mut rng, min_val..=max_val)).collect())
        .collect()
}
