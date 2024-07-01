//! Provides functions for calculating Lp-norms between two vectors.

use core::cmp::Ordering;

use crate::{number::Float, Number};

/// Euclidean distance between two vectors.
///
/// Also known as the L2-norm, the Euclidean distance is defined as the square
/// root of the sum of the squares of the absolute differences between the
/// corresponding elements of the two vectors.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x` - The first slice of `Number`s.
/// * `y` - The second slice of `Number`s.
///
/// # Examples
///
/// ```
/// use distances::vectors::euclidean;
///
/// let x: Vec<f64> = vec![1.0, 2.0, 3.0];
/// let y: Vec<f64> = vec![4.0, 5.0, 6.0];
///
/// let distance: f64 = euclidean(&x, &y);
///
/// assert!((distance - (27.0_f64).sqrt()).abs() <= f64::EPSILON);
/// ```
pub fn euclidean<T: AsRef<[U]>, U: Number, F: Float>(x: T, y: T) -> F {
    euclidean_sq::<T, U, F>(x, y).sqrt()
}

/// Squared Euclidean distance between two vectors.
///
/// Also known as the squared L2-norm, the squared Euclidean distance is defined
/// as the sum of the squares of the absolute differences between the
/// corresponding elements of the two vectors.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x` - The first slice of `Number`s.
/// * `y` - The second slice of `Number`s.
///
/// # Examples
///
/// ```
/// use distances::vectors::euclidean_sq;
///
/// let x: Vec<f64> = vec![1.0, 2.0, 3.0];
/// let y: Vec<f64> = vec![4.0, 5.0, 6.0];
///
/// let distance: f64 = euclidean_sq(&x, &y);
///
/// assert!((distance - 27.0).abs() <= f64::EPSILON);
/// ```
pub fn euclidean_sq<T: AsRef<[U]>, U: Number, F: Number>(x: T, y: T) -> F {
    x.as_ref()
        .iter()
        .zip(y.as_ref().iter())
        .map(|(a, &b)| a.abs_diff(b))
        .map(|v| v * v)
        .map(F::from)
        .sum()
}

/// Manhattan distance between two vectors.
///
/// Also known as the L1-norm or the taxicab distance, the Manhattan distance is
/// defined as the sum of the absolute differences between the corresponding
/// elements of the two vectors.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x` - The first slice of `Number`s.
/// * `y` - The second slice of `Number`s.
///
/// # Examples
///
/// ```
/// use distances::vectors::manhattan;
///
/// let x: Vec<f64> = vec![1.0, 2.0, 3.0];
/// let y: Vec<f64> = vec![4.0, 5.0, 6.0];
///
/// let distance: f64 = manhattan(&x, &y);
///
/// assert!((distance - 9.0).abs() <= f64::EPSILON);
///
/// let distance: f64 = manhattan(x, y);
///
/// assert!((distance - 9.0).abs() <= f64::EPSILON);
/// ```
pub fn manhattan<T: AsRef<[U]>, U: Number>(x: T, y: T) -> U {
    x.as_ref()
        .iter()
        .zip(y.as_ref().iter())
        .map(|(a, &b)| a.abs_diff(b))
        .sum()
}

/// L3-norm between two vectors.
///
/// The L3-norm is defined as the cubic root of the sum of the cubes of the
/// absolute differences between the corresponding elements of the two vectors.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x` - The first slice of `Number`s.
/// * `y` - The second slice of `Number`s.
///
/// # Examples
///
/// ```
/// use distances::vectors::l3_norm;
///
/// let x: Vec<f64> = vec![1.0, 2.0, 3.0];
/// let y: Vec<f64> = vec![4.0, 5.0, 6.0];
///
/// let distance: f64 = l3_norm(&x, &y);
///
/// assert!((distance - (81.0_f64).cbrt()).abs() <= f64::EPSILON);
/// ```
pub fn l3_norm<T: AsRef<[U]>, U: Number, F: Float>(x: T, y: T) -> F {
    x.as_ref()
        .iter()
        .zip(y.as_ref().iter())
        .map(|(a, &b)| a.abs_diff(b))
        .map(F::from)
        .map(|v| v * v * v)
        .sum::<F>()
        .cbrt()
}

/// L4-norm between two vectors.
///
/// The L4-norm is defined as the fourth root of the sum of the fourth powers of
/// the absolute differences between the corresponding elements of the two vectors.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x` - The first slice of `Number`s.
/// * `y` - The second slice of `Number`s.
///
/// # Examples
///
/// ```
/// use distances::vectors::l4_norm;
///
/// let x: Vec<f64> = vec![1.0, 2.0, 3.0];
/// let y: Vec<f64> = vec![4.0, 5.0, 6.0];
///
/// let distance: f64 = l4_norm(&x, &y);
///
/// assert!((distance - (243.0_f64).sqrt().sqrt()).abs() <= f64::EPSILON);
/// ```
pub fn l4_norm<T: AsRef<[U]>, U: Number, F: Float>(x: T, y: T) -> F {
    x.as_ref()
        .iter()
        .zip(y.as_ref().iter())
        .map(|(a, &b)| a.abs_diff(b))
        .map(F::from)
        .map(|v| v * v)
        .map(|v| v * v)
        .sum::<F>()
        .sqrt()
        .sqrt()
}

/// Chebyshev distance between two vectors.
///
/// Also known as the L∞-norm, the Chebyshev distance is defined as the maximum
/// absolute difference between the corresponding elements of the two vectors.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x` - The first slice of `Number`s.
/// * `y` - The second slice of `Number`s.
///
/// # Examples
///
/// ```
/// use distances::vectors::chebyshev;
///
/// let x: Vec<f64> = vec![1.0, 2.0, 3.0];
/// let y: Vec<f64> = vec![6.0, 5.0, 4.0];
///
/// let distance: f64 = chebyshev(&x, &y);
///
/// assert!((distance - 5.0).abs() <= f64::EPSILON);
/// ```
pub fn chebyshev<T: AsRef<[U]>, U: Number>(x: T, y: T) -> U {
    x.as_ref()
        .iter()
        .zip(y.as_ref().iter())
        .map(|(a, &b)| a.abs_diff(b))
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
        .unwrap_or_else(U::zero)
}

/// General (Lp-norm)^p between two vectors.
///
/// This is defined as the sum of the pth powers of the absolute differences
/// between the corresponding elements of the two slices.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x` - The first slice of `Number`s.
/// * `y` - The second slice of `Number`s.
///
/// # Examples
///
/// ```
/// use distances::vectors::minkowski_p;
///
/// let metric = minkowski_p(3);
///
/// let x: Vec<f64> = vec![1.0, 2.0, 3.0];
/// let y: Vec<f64> = vec![4.0, 5.0, 6.0];
///
/// let distance: f64 = metric(&x, &y);
/// assert!((distance - 81.0).abs() <= 1e-12);
/// ```
pub fn minkowski_p<T: AsRef<[U]>, U: Number, F: Float>(p: i32) -> impl Fn(&T, &T) -> F {
    move |x: &T, y: &T| {
        x.as_ref()
            .iter()
            .zip(y.as_ref().iter())
            .map(|(a, &b)| a.abs_diff(b))
            .map(F::from)
            .map(|v| v.powi(p))
            .sum()
    }
}

/// General Lp-norm between two vectors.
///
/// The Lp-norm is defined as the pth root of the sum of the pth powers of
/// the absolute differences between the corresponding elements of the two
/// vectors.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x` - The first slice of `Number`s.
/// * `y` - The second slice of `Number`s.
///
/// # Examples
///
/// ```
/// use distances::vectors::minkowski;
///
/// let metric = minkowski(3);
///
/// let x: Vec<f64> = vec![1.0, 2.0, 3.0];
/// let y: Vec<f64> = vec![4.0, 5.0, 6.0];
///
/// let distance: f64 = metric(&x, &y);
/// assert!((distance - (81.0_f64).cbrt()).abs() <= 1e-12);
/// ```
pub fn minkowski<T: AsRef<[U]>, U: Number, F: Float>(p: i32) -> impl Fn(&T, &T) -> F {
    move |x: &T, y: &T| minkowski_p::<T, U, F>(p)(x, y).powf(F::one() / F::from(p))
}
