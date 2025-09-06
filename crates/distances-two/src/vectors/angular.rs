//! Vector distances based on the angles between vectors.

/// The cosine distance between two vectors.
///
/// This is defined as `1 - cosine_similarity(a, b)`.
pub fn cosine<S, T>(a: &S, b: &S) -> T
where
    S: AsRef<[T]>,
    T: num_traits::Float,
{
    T::one() - cosine_similarity(a, b)
}

/// The cosine distance between two vectors that have been L2-normalized.
///
/// If the vectors have not been normalized, the result will be nonsensical. Use [`cosine`] instead.
pub fn cosine_normalized<S, T>(a: &S, b: &S) -> T
where
    S: AsRef<[T]>,
    T: num_traits::Num + Copy,
{
    T::one() - dot_product(a, b)
}

/// The cosine similarity between two vectors.
///
/// This is defined as the dot product of the vectors divided by the product of their magnitudes.
pub fn cosine_similarity<S, T>(a: &S, b: &S) -> T
where
    S: AsRef<[T]>,
    T: num_traits::Float,
{
    let (ab, aa_sq, bb_sq) = a
        .as_ref()
        .iter()
        .zip(b.as_ref())
        .fold((T::zero(), T::zero(), T::zero()), |(ab, aa_sq, bb_sq), (&x, &y)| {
            (ab + x * y, aa_sq + x * x, bb_sq + y * y)
        });
    if ab.is_zero() { T::zero() } else { ab / (aa_sq * bb_sq).sqrt() }
}

/// The cosine similarity between two vectors that have been L2-normalized.
///
/// If the vectors have not been normalized, the result will be nonsensical. Use [`cosine_similarity`] instead.
pub fn cosine_similarity_normalized<S, T>(a: &S, b: &S) -> T
where
    S: AsRef<[T]>,
    T: num_traits::Num + Copy,
{
    dot_product(a, b)
}

/// The dot-product of two vectors.
pub fn dot_product<S, T>(a: &S, b: &S) -> T
where
    S: AsRef<[T]>,
    T: num_traits::Num + Copy,
{
    a.as_ref().iter().zip(b.as_ref()).map(|(&x, &y)| x * y).fold(T::zero(), |acc, val| acc + val)
}
