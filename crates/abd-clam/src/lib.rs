//! Redesign of CLAM
// #![doc = include_str!("../README.md")]

use core::{
    fmt::{Debug, Display},
    str::FromStr,
};

pub mod cakes;
pub mod codec;
mod tree;
pub mod utils; // Intended for private use, but made public for testing

pub use tree::{BranchingFactor, Cluster, PartitionStrategy, SpanReductionFactor, Tree};

#[cfg(feature = "musals")]
pub mod musals;

/// A trait for types that can be used as distance values in clustering algorithms.
#[must_use]
pub trait DistanceValue:
    PartialEq
    + PartialOrd
    + Copy
    + Display
    + Debug
    + FromStr
    + num_traits::Num
    + num_traits::NumRef
    + num_traits::RefNum<Self>
    + num_traits::NumAssignOps
    + num_traits::NumAssign
    + num_traits::NumAssignRef
    + num_traits::Bounded
    + num_traits::ToPrimitive
    + num_traits::FromPrimitive
    + std::iter::Sum
{
    /// Returns half of the value.
    #[must_use]
    fn half(self) -> Self {
        self / (Self::one() + Self::one())
    }
}

/// Blanket implementation of `DistanceValue` for all types that satisfy the trait bounds.
impl<T> DistanceValue for T where
    T: PartialEq
        + PartialOrd
        + Copy
        + Display
        + Debug
        + FromStr
        + num_traits::Num
        + num_traits::NumRef
        + num_traits::RefNum<Self>
        + num_traits::NumAssignOps
        + num_traits::NumAssign
        + num_traits::NumAssignRef
        + num_traits::Bounded
        + num_traits::ToPrimitive
        + num_traits::FromPrimitive
        + std::iter::Sum
{
}

/// A trait for types that can be used as floating-point distance values in clustering algorithms.
pub trait FloatDistanceValue:
    DistanceValue + num_traits::Float + num_traits::FloatConst + num_traits::Pow<Self, Output = Self>
{
}

impl<T> FloatDistanceValue for T where
    T: DistanceValue + num_traits::Float + num_traits::FloatConst + num_traits::Pow<Self, Output = Self>
{
}

// /// A trait for types that can be used as floating-point distance values in
// /// clustering algorithms.
// pub trait FloatDistanceValue: DistanceValue + num::Float + num::traits::FloatConst {
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
