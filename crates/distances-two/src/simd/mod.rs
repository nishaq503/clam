//! SIMD accelerated distance functions

#[cfg(all(feature = "simd-128", feature = "simd-256", feature = "simd-512"))]
compile_error!("Only one SIMD feature may be active for any given compilation. Found `simd-128`, `simd-256` and `simd-512` active at once.");

#[cfg(all(feature = "simd-128", feature = "simd-256"))]
compile_error!(
    "Only one SIMD feature may be active for any given compilation. Found `simd-128` and `simd-256` active at once."
);

#[cfg(all(feature = "simd-128", feature = "simd-512"))]
compile_error!(
    "Only one SIMD feature may be active for any given compilation. Found `simd-128` and `simd-512` active at once."
);

#[cfg(all(feature = "simd-256", feature = "simd-512"))]
compile_error!(
    "Only one SIMD feature may be active for any given compilation. Found `simd-256` and `simd-512` active at once."
);

#[cfg(feature = "simd-128")]
mod simd_128;

#[cfg(feature = "simd-256")]
mod simd_256;

#[cfg(feature = "simd-512")]
mod simd_512;

#[cfg(feature = "simd-128")]
pub use simd_128::*;

#[cfg(feature = "simd-256")]
pub use simd_256::*;

#[cfg(feature = "simd-512")]
pub use simd_512::*;
