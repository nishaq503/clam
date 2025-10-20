//! SIMD accelerated distance functions

#![expect(dead_code)]

#[cfg(any(
    all(feature = "simd-128", feature = "simd-256", feature = "simd-512"),
    all(feature = "simd-128", feature = "simd-256"),
    all(feature = "simd-128", feature = "simd-512"),
    all(feature = "simd-256", feature = "simd-512"),
))]
compile_error!("Only one of `simd-128`, `simd-256` and `simd-512` features may be active.");

#[macro_use]
mod macros;

#[cfg(feature = "simd-128")]
pub(crate) mod simd_128;

#[cfg(feature = "simd-256")]
pub(crate) mod simd_256;

#[cfg(feature = "simd-512")]
pub(crate) mod simd_512;

mod traits;

pub(crate) use traits::Naive;
pub use traits::SIMD;

/// SIMD accelerated squared Euclidean distance between two vectors.
pub fn euclidean_sq<S: SIMD>(x: S, y: S) -> S::Output {
    x.squared_euclidean(y)
}

/// SIMD accelerated Euclidean distance between two vectors.
pub fn euclidean<S: SIMD>(x: S, y: S) -> S::Output {
    x.euclidean(y)
}
