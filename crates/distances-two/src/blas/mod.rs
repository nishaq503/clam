//! BLAS-accelerated distance functions.

/// This private module contains a sealed trait to prevent downstream implementations of `BlasFloat`.
mod private {
    /// A sealed trait to prevent downstream implementations of `BlasFloat`.
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A trait for floating-point types that can be used with BLAS-accelerated distance functions.
pub trait BlasFloat: private::Sealed + num_traits::Float + rust_blas::Copy + rust_blas::Axpy + rust_blas::Nrm2 + rust_blas::Dot {}

impl BlasFloat for f32 {}
impl BlasFloat for f64 {}

/// The Cosine distance between two vectors using BLAS.
pub fn cosine<S, T>(x: &S, y: &S) -> T
where
    S: AsRef<[T]>,
    T: BlasFloat,
{
    T::one() - cosine_similarity(x, y)
}

/// The cosine distance between two vectors that have unit L2-norms using BLAS.
pub fn cosine_normalized<S, T>(x: &S, y: &S) -> T
where
    S: AsRef<[T]>,
    T: BlasFloat,
{
    let (x, y) = (x.as_ref(), y.as_ref());
    T::one() - rust_blas::Dot::dot(x, y)
}

/// The Cosine similarity between two vectors using BLAS.
pub fn cosine_similarity<S, T>(x: &S, y: &S) -> T
where
    S: AsRef<[T]>,
    T: BlasFloat,
{
    let (x, y) = (x.as_ref(), y.as_ref());
    let xy = rust_blas::Dot::dot(x, y);
    if xy.is_zero() {
        T::zero()
    } else {
        let xx = rust_blas::Nrm2::nrm2(x);
        let yy = rust_blas::Nrm2::nrm2(y);
        let denom = xx * yy;
        if denom.is_zero() { T::zero() } else { xy / denom }
    }
}

/// The cosine similarity between two vectors that have unit L2-norms using BLAS.
pub fn cosine_similarity_normalized<S, T>(x: &S, y: &S) -> T
where
    S: AsRef<[T]>,
    T: BlasFloat,
{
    let (x, y) = (x.as_ref(), y.as_ref());
    rust_blas::Dot::dot(x, y)
}

/// The dot product between two vectors using BLAS.
pub fn dot_product<S, T>(x: &S, y: &S) -> T
where
    S: AsRef<[T]>,
    T: BlasFloat,
{
    let (x, y) = (x.as_ref(), y.as_ref());
    rust_blas::Dot::dot(x, y)
}

/// The Euclidean distance between two vectors using BLAS.
pub fn euclidean<S, T>(x: &S, y: &S) -> T
where
    S: AsRef<[T]>,
    T: BlasFloat,
{
    let (x, y) = (x.as_ref(), y.as_ref());
    let mut diff = vec![T::zero(); x.len()];
    rust_blas::Copy::copy(x, &mut diff);
    rust_blas::Axpy::axpy(&-T::one(), y, &mut diff);
    rust_blas::Nrm2::nrm2(&diff)
}

/// The Squared Euclidean distance between two vectors using BLAS.
pub fn euclidean_sq<S, T>(x: &S, y: &S) -> T
where
    S: AsRef<[T]>,
    T: BlasFloat,
{
    euclidean(x, y).powi(2)
}

/// The L2-norm of a vector using BLAS.
pub fn norm_l2<S, T>(x: &S) -> T
where
    S: AsRef<[T]>,
    T: BlasFloat,
{
    rust_blas::Nrm2::nrm2(x.as_ref())
}
