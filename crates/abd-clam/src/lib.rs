//! Redesign of CLAM
// #![doc = include_str!("../README.md")]

pub mod cakes;
pub mod cluster;
pub mod utils; // Intended for private use, but made public for testing purposes

pub use cluster::Cluster;

/// A trait for types that can be used as distance values in clustering algorithms.
pub trait DistanceValue:
    num::Num
    + num::Bounded
    + num::ToPrimitive
    + num::FromPrimitive
    + num::traits::NumAssignOps
    + std::iter::Sum
    + PartialOrd
    + Copy
{
}

/// Blanket implementation of `DistanceValue` for all types that satisfy the
/// trait bounds.
impl<T> DistanceValue for T where
    T: num::Num
        + num::Bounded
        + num::ToPrimitive
        + num::FromPrimitive
        + num::traits::NumAssignOps
        + std::iter::Sum
        + PartialOrd
        + Copy
{
}

/// A trait for types that can be used as floating-point distance values in
/// clustering algorithms.
pub trait FloatDistanceValue: DistanceValue + num::Float {}

impl<T> FloatDistanceValue for T where T: DistanceValue + num::Float {}

// /// A trait for types that can be used as floating-point distance values in
// /// clustering algorithms.
// pub trait FloatDistanceValue: DistanceValue + num::Float {
//     /// The gauss error function.
//     ///
//     /// The `libm` crate is used to provide the implementations for `f32` and `f64`.
//     #[must_use]
//     fn erf(self) -> Self;
// }

// /// Implementation of `FloatDistanceValue` for `f32`
// impl FloatDistanceValue for f32 {
//     fn erf(self) -> Self {
//         libm::erff(self)
//     }
// }

// /// Implementation of `FloatDistanceValue` for `f64`
// impl FloatDistanceValue for f64 {
//     fn erf(self) -> Self {
//         libm::erf(self)
//     }
// }
