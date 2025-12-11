//! Fast and generic distance functions for high-dimensional data.

#![cfg_attr(
    any(feature = "simd-128", feature = "simd-256", feature = "simd-512", feature = "simd-1024"),
    feature(portable_simd)
)]

#[cfg(feature = "blas")]
pub mod blas;

#[cfg(any(feature = "simd-128", feature = "simd-256", feature = "simd-512", feature = "simd-1024"))]
pub mod simd;

#[cfg(any(feature = "simd-128", feature = "simd-256", feature = "simd-512", feature = "simd-1024"))]
pub mod std_simd;

pub mod vectors;
