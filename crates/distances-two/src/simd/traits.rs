//! Traits for SIMD types and distance functions

/// This module defines the `Sealed` trait to prevent external implementations of `Naive` and `SIMD`.
mod private {
    /// A sealed trait to prevent external implementations of `Naive` and `SIMD`.
    pub trait Sealed {}

    impl Sealed for &[f32] {}
    impl Sealed for &[f64] {}
    impl Sealed for &Vec<f32> {}
    impl Sealed for &Vec<f64> {}
}

/// Trait for naive distance functions.
pub trait Naive: private::Sealed {
    /// The output type of the distance functions.
    type Output;

    /// Squared Euclidean distance between two vectors.
    fn squared_euclidean(self, other: Self) -> Self::Output;

    /// Dot product between two vectors.
    fn dot_product(self, other: Self) -> Self::Output;
}

impl_naive!(f32);
impl_naive!(f64);

/// Trait for SIMD accelerated distance functions.
pub trait SIMD: private::Sealed {
    /// The output type of the distance functions.
    type Output;

    /// Squared Euclidean distance between two vectors.
    fn squared_euclidean(self, other: Self) -> Self::Output;

    /// Euclidean distance between two vectors.
    fn euclidean(self, other: Self) -> Self::Output;

    /// Dot product between two vectors.
    fn dot_product(self, other: Self) -> Self::Output;
}
