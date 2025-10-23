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

/// SIMD accelerated Squared L2 norm of a vector.
pub fn norm_l2_sq<S: Simd>(x: S) -> S::Inner {
    x.norm_l2_sq()
}

/// SIMD accelerated L2 norm of a vector.
pub fn norm_l2<S: Simd>(x: S) -> S::Inner {
    x.norm_l2()
}

/// SIMD accelerated Cosine distance between two vectors.
pub fn cosine<S: Simd>(x: S, y: S) -> S::Inner {
    x.cosine(y)
}

/// SIMD accelerated Cosine similarity between two vectors.
pub fn cosine_similarity<S: Simd>(x: S, y: S) -> S::Inner {
    x.cosine_similarity(y)
}

/// SIMD accelerated Cosine distance between two vectors that have unit L2 norm.
pub fn cosine_normalized<S: Simd>(x: S, y: S) -> S::Inner {
    x.cosine_normalized(y)
}

/// SIMD accelerated Cosine similarity between two vectors that have unit L2 norm.
pub fn cosine_similarity_normalized<S: Simd>(x: S, y: S) -> S::Inner {
    x.cosine_similarity_normalized(y)
}
