//! Clustering, Learning, and Approximation with Manifolds.
//!
//! We provide functionality for clustering, search, multiple sequence alignment, anomaly detection, compression, compressive search, dimension reduction, and
//! more. All algorithms are designed to work efficiently with large high-dimensional datasets under arbitrary distance functions.
//!
//! # Modules and Features
//!
//! - [`Tree`]: Hierarchical clustering algorithms and data structures.
//! - [`cakes`]: Search (k-NN, range search) algorithms.
//! - [`musals`]: Multiple sequence alignment of genomic and protein sequences. Use the `musals` feature to enable this module.
//! - `chaoda`: Anomaly detection algorithms using clustering trees and graphs. Use the `chaoda` feature to enable this module. WIP.
//! - `codec`: Compression and compressive search algorithms. Use the `codec` feature to enable this module. WIP.
//! - `mbed`: Dimension reduction algorithms. Use the `mbed` feature to enable this module. WIP.

use core::{
    fmt::{Debug, Display},
    str::FromStr,
};

pub mod cakes;
mod tree;
pub mod utils; // Intended for private use, but made public for testing

pub use tree::{BranchingFactor, Cluster, PartitionStrategy, SpanReductionFactor, Tree};

#[cfg(feature = "musals")]
pub mod musals;

/// A trait for types that can be used as distance values in clustering algorithms.
///
/// We provide a blanket implementation for all types that satisfy the trait bounds. This includes all primitive numeric types.
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
///
/// We provide a blanket implementation for all types that satisfy the trait bounds. This includes all primitive float types.
pub trait FloatDistanceValue: DistanceValue + num_traits::Float + num_traits::FloatConst + num_traits::Pow<Self, Output = Self> {}

impl<T> FloatDistanceValue for T where T: DistanceValue + num_traits::Float + num_traits::FloatConst + num_traits::Pow<Self, Output = Self> {}
