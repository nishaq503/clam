//! SIMD accelerated distance functions

#![expect(dead_code)]

/// Ensure that only one SIMD feature is enabled at a time.
macro_rules! assert_unique_feature {
    () => {};
    ($first:tt $(,$rest:tt)*) => {
        $(
            #[cfg(all(feature = $first, feature = $rest))]
            compile_error!(concat!("features \"", $first, "\" and \"", $rest, "\" cannot be used together"));
        )*
        assert_unique_feature!($($rest),*);
    }
}
assert_unique_feature!("simd-128", "simd-256", "simd-512", "simd-1024");

#[macro_use]
mod macros;

pub(crate) use macros::simd::Simd;

#[cfg(feature = "simd-128")]
pub(crate) mod simd_128;

#[cfg(feature = "simd-256")]
pub(crate) mod simd_256;

#[cfg(feature = "simd-512")]
pub(crate) mod simd_512;

#[cfg(feature = "simd-1024")]
pub(crate) mod simd_1024;

/// SIMD accelerated squared Euclidean distance between two vectors.
pub fn euclidean_sq<S: Simd>(x: S, y: S) -> S::Inner {
    x.squared_euclidean(y)
}

/// SIMD accelerated Euclidean distance between two vectors.
pub fn euclidean<S: Simd>(x: S, y: S) -> S::Inner {
    x.euclidean(y)
}

/// SIMD accelerated dot product between two vectors.
pub fn dot_product<S: Simd>(x: S, y: S) -> S::Inner {
    x.dot_product(y)
}
