//! Measuring the quality of multiple sequence alignments (MSAs) using various metrics.

#![expect(clippy::cast_precision_loss)]

use crate::{DistanceValue, Tree};

use super::Sequence;

mod gap_fraction;
mod mismatch_fraction;

pub use gap_fraction::GapFraction;
pub use mismatch_fraction::MismatchFraction;

/// An enumeration of the quality metrics that can be computed for an MSA.
pub enum QualityMetric {
    /// The mean fraction of gaps in the sequences of the MSA.
    GapFraction,
    /// The fraction of mismatches between pairs of sequences in the MSA.
    MismatchFraction,
}

/// An enumeration of the quality metrics that have been computed for an MSA.
pub enum QualityMetricResult {
    /// The mean fraction of gaps in the sequences of the MSA.
    GapFraction(GapFraction),
    /// The fraction of mismatches between pairs of sequences in the MSA.
    MismatchFraction(MismatchFraction),
}

impl QualityMetricResult {
    /// Returns the name of the quality metric.
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Self::GapFraction(metric) => metric.name(),
            Self::MismatchFraction(metric) => metric.name(),
        }
    }

    /// Returns a description of the quality metric.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::GapFraction(metric) => metric.description(),
            Self::MismatchFraction(metric) => metric.description(),
        }
    }

    /// Returns the mean value of the quality metric.
    #[must_use]
    pub fn mean(&self) -> f64 {
        match self {
            Self::GapFraction(metric) => metric.mean(),
            Self::MismatchFraction(metric) => metric.mean(),
        }
    }

    /// Returns the standard deviation of the quality metric.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        match self {
            Self::GapFraction(metric) => metric.std_dev(),
            Self::MismatchFraction(metric) => metric.std_dev(),
        }
    }

    /// Returns the minimum value of the quality metric.
    #[must_use]
    pub fn min(&self) -> f64 {
        match self {
            Self::GapFraction(metric) => metric.min(),
            Self::MismatchFraction(metric) => metric.min(),
        }
    }

    /// Returns the maximum value of the quality metric.
    #[must_use]
    pub fn max(&self) -> f64 {
        match self {
            Self::GapFraction(metric) => metric.max(),
            Self::MismatchFraction(metric) => metric.max(),
        }
    }
}

/// A trait for quality metrics that can be computed from a multiple sequence alignment (MSA).
pub trait MsaQuality {
    /// Returns the name of the quality metric.
    fn name(&self) -> String;

    /// Returns a description of the quality metric.
    fn description(&self) -> String;

    /// Returns the mean value of the quality metric for the MSA. Ideally, this would be a getter method.
    fn mean(&self) -> f64;

    /// Returns the standard deviation of the quality metric for the MSA. Ideally, this would be a getter method.
    fn std_dev(&self) -> f64;

    /// Returns the minimum value of the quality metric for the MSA. Ideally, this would be a getter method.
    fn min(&self) -> f64;

    /// Returns the maximum value of the quality metric for the MSA. Ideally, this would be a getter method.
    fn max(&self) -> f64;

    /// Computes the quality metric from the given MSA tree.
    fn compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>) -> Self
    where
        S: Sequence,
        T: DistanceValue,
        M: Fn(&S, &S) -> T,
        Self: Sized;

    /// Parallel version of [`compute`].
    fn par_compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>) -> Self
    where
        Id: Send + Sync,
        S: Sequence + Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Fn(&S, &S) -> T + Send + Sync,
        Self: Sized + Send + Sync;
}

/// Computes the mean, standard deviation, minimum, and maximum of a slice of f64 values.
fn mu_sigma_min_max<I: AsRef<[f64]>>(values: I) -> (f64, f64, f64, f64) {
    let values = values.as_ref();

    let [sum, min, max] = values
        .iter()
        .fold([0.0, f64::INFINITY, f64::NEG_INFINITY], |[mean, min, max], &x| {
            [mean + x, min.min(x), max.max(x)]
        });

    let num_pairs = values.len() as f64;
    let mean = sum / num_pairs;
    let std_dev = (values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / num_pairs).sqrt();

    (mean, std_dev, min, max)
}
