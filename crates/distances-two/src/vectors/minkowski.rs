//! Lp-norms for vectors.

/// Chebyshev distance between two vectors.
///
/// Also known as the L∞-norm, the Chebyshev distance is defined as the maximum absolute difference between corresponding elements.
///
/// This implementation considers the distance between two empty vectors to be zero, and treats incompatible values as equal to other incompatible values.
pub fn chebyshev<S, T>(x: &S, y: &S) -> T
where
    S: AsRef<[T]>,
    T: num_traits::Num + PartialOrd + Copy,
{
    x.as_ref()
        .iter()
        .zip(y.as_ref())
        .map(|(&a, &b)| if a >= b { a - b } else { b - a })
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
        .unwrap_or_else(T::zero)
}

/// Euclidean distance between two vectors.
///
/// Also known as the L2-norm, the Euclidean distance is defined as the square-root of the sum of the squared differences between corresponding elements.
pub fn euclidean<S, T>(x: &S, y: &S) -> T
where
    S: AsRef<[T]>,
    T: num_traits::Float,
{
    euclidean_sq(x, y).sqrt()
}

/// Squared Euclidean distance between two vectors.
///
/// Also known as the squared L2-norm, the squared Euclidean distance is defined as the sum of the squared differences between corresponding elements.
pub fn euclidean_sq<S, T>(x: &S, y: &S) -> T
where
    S: AsRef<[T]>,
    T: num_traits::Num + PartialOrd + Copy,
{
    x.as_ref()
        .iter()
        .zip(y.as_ref())
        .map(|(&a, &b)| if a >= b { a - b } else { b - a })
        .map(|d| d * d)
        .fold(T::zero(), |acc, v| acc + v)
}

/// Manhattan distance between two vectors.
///
/// Also known as the L1-norm, taxicab or city-block distance, the Manhattan distance is defined as the sum of the absolute differences between corresponding
/// elements.
pub fn manhattan<S, T>(x: &S, y: &S) -> T
where
    S: AsRef<[T]>,
    T: num_traits::Num + PartialOrd + Copy,
{
    x.as_ref()
        .iter()
        .zip(y.as_ref())
        .map(|(&a, &b)| if a >= b { a - b } else { b - a })
        .fold(T::zero(), |acc, v| acc + v)
}

/// Generic Minkowski distance between two vectors.
///
/// The generic Minkowski distance is defined as the p-th root of the sum of the absolute differences between corresponding elements raised to the power of p.
pub fn minkowski<S, T>(x: &S, y: &S, p: T) -> T
where
    S: AsRef<[T]>,
    T: num_traits::Float,
{
    x.as_ref()
        .iter()
        .zip(y.as_ref())
        .map(|(&a, &b)| a - b)
        .map(T::abs)
        .map(|d| d.powf(p))
        .fold(T::zero(), |acc, v| acc + v)
        .powf(p.recip())
}

/// Squared L2-norm (Squared Euclidean norm) of a vector.
pub fn norm_l2_sq<S, T>(x: &S) -> T
where
    S: AsRef<[T]>,
    T: num_traits::Float,
{
    x.as_ref().iter().map(|&a| a * a).fold(T::zero(), |acc, v| acc + v)
}

/// L2-norm (Euclidean norm) of a vector.
pub fn norm_l2<S, T>(x: &S) -> T
where
    S: AsRef<[T]>,
    T: num_traits::Float,
{
    norm_l2_sq(x).sqrt()
}
