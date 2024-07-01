//! Angular distances between vectors.

use crate::{
    number::{Float, Int, UInt},
    Number,
};

/// Computes the Cosine distance between two vectors.
///
/// The cosine distance is defined as `1.0 - c` where `c` is the cosine
/// similarity.
///
/// The cosine similarity is defined as the dot product of the two vectors
/// divided by the product of their magnitudes.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x`: A slice of numbers.
/// * `y`: A slice of numbers.
///
/// # Examples
///
/// ```
/// use distances::vectors::cosine;
///
/// let x: Vec<f32> = vec![1.0, 0.0, 0.0];
/// let y: Vec<f32> = vec![0.0, 1.0, 0.0];
///
/// let distance: f32 = cosine(&x, &y);
///
/// assert!((distance - 1.0).abs() < f32::EPSILON);
/// ```
///
/// # References
///
/// * [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
pub fn cosine<T: AsRef<[U]>, U: Number, F: Float>(x: T, y: T) -> F {
    let [xx, yy, xy] = x
        .as_ref()
        .iter()
        .zip(y.as_ref().iter())
        .fold([U::zero(); 3], |[xx, yy, xy], (&a, &b)| {
            [a.mul_add(a, xx), b.mul_add(b, yy), a.mul_add(b, xy)]
        });
    let [xx, yy, xy] = [F::from(xx), F::from(yy), F::from(xy)];

    if xx < F::epsilon() || yy < F::epsilon() || xy < F::epsilon() {
        F::one()
    } else {
        let d = F::one() - xy * (xx * yy).inv_sqrt();
        if d < F::epsilon() {
            F::zero()
        } else {
            d
        }
    }
}

/// Computes the Hamming distance between two vectors.
///
/// The Hamming distance is defined as the number of positions at which
/// the corresponding elements are different.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x`: A slice of numbers.
/// * `y`: A slice of numbers.
///
/// # Examples
///
/// ```
/// use distances::vectors::hamming;
///
/// let x: Vec<u8> = vec![1, 2, 3];
/// let y: Vec<u8> = vec![1, 2, 3];
///
/// let distance: u8 = hamming(&x, &y);
///
/// assert_eq!(distance, 0);
///
/// let x: Vec<u8> = vec![1, 2, 3];
/// let y: Vec<u8> = vec![1, 2, 4];
///
/// let distance: u8 = hamming(&x, &y);
///
/// assert_eq!(distance, 1);
/// ```
///
/// # References
///
/// * [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance)
pub fn hamming<P: AsRef<[T]>, T: Int, U: UInt>(x: P, y: P) -> U {
    U::from(
        x.as_ref()
            .iter()
            .zip(y.as_ref().iter())
            .filter(|(&a, &b)| a != b)
            .count(),
    )
}

/// Computes the Canberra distance between two vectors.
///
/// The Canberra distance is defined as the sum of the absolute differences
/// between the elements of the two vectors divided by the sum of the absolute
/// values of the elements of the two vectors.
///
/// See the [`crate::vectors`] module documentation for information on this
/// function's potentially unexpected behaviors
///
/// # Arguments
///
/// * `x`: A slice of numbers.
/// * `y`: A slice of numbers.
///
/// # Examples
///
/// ```
/// use distances::vectors::canberra;
///
/// let x: Vec<f32> = vec![1.0, 2.0, 3.0];
/// let y: Vec<f32> = vec![4.0, 5.0, 6.0];
///
/// let distance: f32 = canberra(&x, &y);
///
/// assert!((distance - 143.0 / 105.0).abs() <= f32::EPSILON);
/// ```
///
/// # References
///
/// * [Canberra distance](https://en.wikipedia.org/wiki/Canberra_distance)
pub fn canberra<P: AsRef<[T]>, T: Number, U: Float>(x: P, y: P) -> U {
    x.as_ref()
        .iter()
        .map(|&v| U::from(v))
        .zip(y.as_ref().iter().map(|&v| U::from(v)))
        .map(|(a, b)| a.abs_diff(b) / (a.abs() + b.abs()))
        .fold(U::zero(), |acc, v| acc + v)
}

/// Computes the Bray-Curtis distance between two vectors.
///
/// # Arguments
///
/// * `x`: A slice of numbers.
/// * `y`: A slice of numbers.
///
///
/// # Examples
/// ```
/// use distances::vectors::bray_curtis;
///
/// let x: Vec<usize>  = vec![6, 7, 4];
/// let y: Vec<usize> = vec![10, 0, 6];
///
/// let distance: f32 =  bray_curtis(&x, &y);
///
/// assert!((distance - 13.0 / 33.0).abs() <= f32::EPSILON);
/// ```
///
/// # References
///
/// * [Bray-Curtis Distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html#scipy.spatial.distance.braycurtis)
pub fn bray_curtis<P: AsRef<[T]>, T: Number, U: Float>(x: P, y: P) -> U {
    let [numerator, denominator] = x
        .as_ref()
        .iter()
        .zip(y.as_ref().iter())
        .fold([T::zero(); 2], |[n, d], (&a, &b)| {
            [n + a.abs_diff(b), d + (a + b).abs()]
        });

    if denominator <= numerator {
        U::zero()
    } else {
        U::from(numerator) / U::from(denominator)
    }
}
