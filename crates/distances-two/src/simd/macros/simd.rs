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
                    .map(|(a, b)| {
                        let diff = a - b;
                        diff * diff
                    })
                    .fold(<$outer>::splat(0.0), |acc, x| acc + x)
                    .horizontal_add();

                let rem = crate::vectors::euclidean_sq(&a_chunks.remainder(), &b_chunks.remainder());

                sum + rem
            }

            fn euclidean(self, other: Self) -> Self::Inner {
                crate::simd::Simd::squared_euclidean(self, other).sqrt()
            }

            fn dot_product(self, other: Self) -> Self::Inner {
                debug_assert_eq!(self.len(), other.len());

                let mut a_chunks = self.chunks_exact(<$outer>::lanes());
                let mut b_chunks = other.chunks_exact(<$outer>::lanes());

                let sum = a_chunks
                    .by_ref()
                    .map(<$outer>::from_slice)
                    .zip(b_chunks.by_ref().map(<$outer>::from_slice))
                    .map(|(a, b)| a * b)
                    .fold(<$outer>::splat(0.0), |acc, x| acc + x)
                    .horizontal_add();

                let rem = crate::vectors::dot_product(&a_chunks.remainder(), &b_chunks.remainder());

                sum + rem
            }
        }
    };
}
