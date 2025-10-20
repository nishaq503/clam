//! Naive implementations of various distance metrics for testing purposes.

#![allow(dead_code)]

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
    iter_abs_diff(x, y)
        .map(|d| d.powi(4))
        .fold(F::zero(), |acc, d4| acc + d4)
        .sqrt()
        .sqrt()
}

/// Computes the Chebyshev (L∞) distance between two vectors.
pub fn l_inf<F: num_traits::Float>(x: &[F], y: &[F]) -> F {
    iter_abs_diff(x, y).fold(F::zero(), |acc, d| if d > acc { d } else { acc })
}
