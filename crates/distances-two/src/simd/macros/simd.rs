//! Macros for implementing SIMD distance functions

/// This module defines the `Sealed` trait to prevent external implementations of `Naive` and `SIMD`.
mod private {
    /// A sealed trait to prevent external implementations of `Naive` and `SIMD`.
    pub trait Sealed {}

    impl Sealed for &[f32] {}
    impl Sealed for &[f64] {}
    impl Sealed for &Vec<f32> {}
    impl Sealed for &Vec<f64> {}
}

/// Trait for SIMD accelerated distance functions.
pub trait Simd: private::Sealed {
    /// The output type of the distance functions.
    type Inner;

    /// Squared Euclidean distance between two vectors.
    fn squared_euclidean(self, other: Self) -> Self::Inner;

    /// Euclidean distance between two vectors.
    fn euclidean(self, other: Self) -> Self::Inner;

    /// Dot product between two vectors.
    fn dot_product(self, other: Self) -> Self::Inner;

    /// Squared L2 norm of a vector.
    fn norm_l2_sq(self) -> Self::Inner;

    /// L2 norm of a vector.
    fn norm_l2(self) -> Self::Inner;
}

/// Macro to implement the SIMD trait for a given SIMD type, underlying scalar type, and array type
macro_rules! impl_simd {
    ($outer:ty, $inner:ty, $arr:ty) => {
        impl crate::simd::Simd for $arr {
            type Inner = $inner;

            fn squared_euclidean(self, other: Self) -> Self::Inner {
                debug_assert_eq!(self.len(), other.len());

                let mut a_chunks = self.chunks_exact(<$outer>::lanes());
                let mut b_chunks = other.chunks_exact(<$outer>::lanes());

                let sum = a_chunks
                    .by_ref()
                    .map(<$outer>::from_slice)
                    .zip(b_chunks.by_ref().map(<$outer>::from_slice))
                    .map(|(a, b)| a - b)
                    .fold(<$outer>::splat(0.0), |acc, diff| diff * diff + acc)
                    .horizontal_add();

                let rem = crate::vectors::euclidean_sq(&a_chunks.remainder(), &b_chunks.remainder());

                sum + rem
            }

            fn euclidean(self, other: Self) -> Self::Inner {
                self.squared_euclidean(other).sqrt()
            }

            fn dot_product(self, other: Self) -> Self::Inner {
                debug_assert_eq!(self.len(), other.len());

                let mut a_chunks = self.chunks_exact(<$outer>::lanes());
                let mut b_chunks = other.chunks_exact(<$outer>::lanes());

                let sum = a_chunks
                    .by_ref()
                    .map(<$outer>::from_slice)
                    .zip(b_chunks.by_ref().map(<$outer>::from_slice))
                    .fold(<$outer>::splat(0.0), |acc, (a, b)| a * b + acc)
                    .horizontal_add();

                let rem = crate::vectors::dot_product(&a_chunks.remainder(), &b_chunks.remainder());

                sum + rem
            }

            fn norm_l2_sq(self) -> Self::Inner {
                let mut chunks = self.chunks_exact(<$outer>::lanes());

                let sum = chunks
                    .by_ref()
                    .map(<$outer>::from_slice)
                    .fold(<$outer>::splat(0.0), |acc, a| a * a + acc)
                    .horizontal_add();

                let rem = crate::vectors::norm_l2_sq(&chunks.remainder());

                sum + rem
            }

            fn norm_l2(self) -> Self::Inner {
                self.norm_l2_sq().sqrt()
            }
        }
    };
}
