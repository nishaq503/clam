//! Fast and generic distance functions for high-dimensional data.

#[cfg(feature = "blas")]
pub mod blas;

#[cfg(any(
    feature = "simd-128",
    feature = "simd-256",
    feature = "simd-512",
    feature = "simd-1024"
))]
pub mod simd;

pub mod vectors;
