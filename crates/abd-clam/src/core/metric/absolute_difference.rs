//! The `AbsoluteDifference` metric.

use distances::Number;

use super::{Metric, ParMetric};

/// The `AbsoluteDifference` metric measures the absolute difference between two
/// values. It is meant to be used with scalars.
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Deserialize, serde::Serialize)
)]
pub struct AbsoluteDifference;

impl<T: Number> Metric<T, T> for AbsoluteDifference {
    fn distance(&self, a: &T, b: &T) -> T {
        a.abs_diff(*b)
    }

    fn name(&self) -> &'static str {
        "absolute-difference"
    }

    fn has_identity(&self) -> bool {
        true
    }

    fn has_non_negativity(&self) -> bool {
        true
    }

    fn has_symmetry(&self) -> bool {
        true
    }

    fn obeys_triangle_inequality(&self) -> bool {
        true
    }

    fn is_expensive(&self) -> bool {
        false
    }
}

impl<T: Number> ParMetric<T, T> for AbsoluteDifference {}
