//! Measuring the quality of multiple sequence alignments (MSAs) using various metrics.

#![expect(clippy::cast_precision_loss)]

use rand::prelude::*;

use crate::{DistanceValue, Tree};

use super::{CostMatrix, Sequence};

mod distance_distortion;
mod gap_fraction;
mod mismatch_fraction;
mod pairwise_scores;
mod sum_of_pairs;
mod weighted_pairwise_scores;

pub use distance_distortion::DistanceDistortion;
pub use gap_fraction::GapFraction;
pub use mismatch_fraction::MismatchFraction;
pub use pairwise_scores::PairwiseScores;
pub use sum_of_pairs::SumOfPairs;
pub use weighted_pairwise_scores::WeightedPairwiseScores;

/// Quality metrics that may be computed for an MSA.
///
/// Once computed, these will return a [`QualityMetricResult`].
#[non_exhaustive]
pub enum QualityMetric {
    /// The mean fraction of gaps in the sequences of the MSA.
    GapFraction,
    /// The fraction of mismatches between pairs of sequences in the MSA.
    MismatchFraction,
    /// The scores of pairwise alignments in the MSA.
    PairwiseScores,
    /// The scores of weighted pairwise alignments in the MSA.
    WeightedPairwiseScores,
    /// The mean distortion of alignment distances between pairs of sequences in the MSA.
    DistanceDistortion,
    /// The Sum of Pairs (SP) score of the MSA.
    SumOfPairs,
}

/// Quality metrics that have been computed for an MSA.
///
/// Each variant has a name and description, and provides methods to access the mean, standard deviation, minimum, and maximum values of the evaluated metric.
#[non_exhaustive]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum QualityMetricResult {
    /// The mean fraction of gaps in the sequences of the MSA.
    GapFraction(GapFraction),
    /// The fraction of mismatches between pairs of sequences in the MSA.
    MismatchFraction(MismatchFraction),
    /// The scores of pairwise alignments in the MSA.
    PairwiseScores(PairwiseScores),
    /// The scores of weighted pairwise alignments in the MSA.
    WeightedPairwiseScores(WeightedPairwiseScores),
    /// The mean distortion of alignment distances between pairs of sequences in the MSA.
    DistanceDistortion(DistanceDistortion),
    /// The Sum of Pairs (SP) score of the MSA.
    SumOfPairs(SumOfPairs),
}

impl QualityMetricResult {
    /// Returns the name of the quality metric.
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Self::GapFraction(metric) => metric.name(),
            Self::MismatchFraction(metric) => metric.name(),
            Self::PairwiseScores(metric) => metric.name(),
            Self::WeightedPairwiseScores(metric) => metric.name(),
            Self::DistanceDistortion(metric) => metric.name(),
            Self::SumOfPairs(metric) => metric.name(),
        }
    }

    /// Returns a short name of the quality metric, usually an acronym for use as a suffix in file names.
    #[must_use]
    pub fn short_name<'a>(&self) -> &'a str {
        match self {
            Self::GapFraction(metric) => metric.short_name(),
            Self::MismatchFraction(metric) => metric.short_name(),
            Self::PairwiseScores(metric) => metric.short_name(),
            Self::WeightedPairwiseScores(metric) => metric.short_name(),
            Self::DistanceDistortion(metric) => metric.short_name(),
            Self::SumOfPairs(metric) => metric.short_name(),
        }
    }

    /// Returns a description of the quality metric.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::GapFraction(metric) => metric.description(),
            Self::MismatchFraction(metric) => metric.description(),
            Self::PairwiseScores(metric) => metric.description(),
            Self::WeightedPairwiseScores(metric) => metric.description(),
            Self::DistanceDistortion(metric) => metric.description(),
            Self::SumOfPairs(metric) => metric.description(),
        }
    }

    /// Returns the mean value of the quality metric.
    #[must_use]
    pub fn mean(&self) -> f64 {
        match self {
            Self::GapFraction(metric) => metric.mean(),
            Self::MismatchFraction(metric) => metric.mean(),
            Self::PairwiseScores(metric) => metric.mean(),
            Self::WeightedPairwiseScores(metric) => metric.mean(),
            Self::DistanceDistortion(metric) => metric.mean(),
            Self::SumOfPairs(metric) => metric.mean(),
        }
    }

    /// Returns the standard deviation of the quality metric.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        match self {
            Self::GapFraction(metric) => metric.std_dev(),
            Self::MismatchFraction(metric) => metric.std_dev(),
            Self::PairwiseScores(metric) => metric.std_dev(),
            Self::WeightedPairwiseScores(metric) => metric.std_dev(),
            Self::DistanceDistortion(metric) => metric.std_dev(),
            Self::SumOfPairs(metric) => metric.std_dev(),
        }
    }

    /// Returns the minimum value of the quality metric.
    #[must_use]
    pub fn min(&self) -> f64 {
        match self {
            Self::GapFraction(metric) => metric.min(),
            Self::MismatchFraction(metric) => metric.min(),
            Self::PairwiseScores(metric) => metric.min(),
            Self::WeightedPairwiseScores(metric) => metric.min(),
            Self::DistanceDistortion(metric) => metric.min(),
            Self::SumOfPairs(metric) => metric.min(),
        }
    }

    /// Returns the maximum value of the quality metric.
    #[must_use]
    pub fn max(&self) -> f64 {
        match self {
            Self::GapFraction(metric) => metric.max(),
            Self::MismatchFraction(metric) => metric.max(),
            Self::PairwiseScores(metric) => metric.max(),
            Self::WeightedPairwiseScores(metric) => metric.max(),
            Self::DistanceDistortion(metric) => metric.max(),
            Self::SumOfPairs(metric) => metric.max(),
        }
    }
}

/// A trait for quality metrics that can be computed from a multiple sequence alignment (MSA).
pub trait MsaQuality: serde::Serialize + for<'de> serde::Deserialize<'de> {
    /// Returns the name of the quality metric.
    fn name(&self) -> String;

    /// Returns a short name of the quality metric, usually an acronym for use as a suffix in file names.
    fn short_name<'a>(&self) -> &'a str;

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
    fn compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>, cost_matrix: &CostMatrix<T>, sample_size: Option<usize>) -> Self
    where
        S: Sequence,
        T: DistanceValue,
        M: Fn(&S, &S) -> T,
        Self: Sized;

    /// Parallel version of [`compute`].
    fn par_compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>, cost_matrix: &CostMatrix<T>, sample_size: Option<usize>) -> Self
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

    let [sum, min, max] = values.iter().fold([0.0, f64::INFINITY, f64::NEG_INFINITY], |[mean, min, max], &x| {
        [mean + x, min.min(x), max.max(x)]
    });

    let num_pairs = values.len() as f64;
    let mean = sum / num_pairs;
    let std_dev = (values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / num_pairs).sqrt();

    (mean, std_dev, min, max)
}

/// Returns a random sample of indices from 0 to `max_index - 1`.
fn random_sample_indices(max_index: usize, sample_size: Option<usize>) -> Vec<usize> {
    let mut indices = (0..max_index).collect::<Vec<_>>();

    if let Some(size) = sample_size {
        let mut rng = rand::rng();
        indices.shuffle(&mut rng);
        indices.truncate(size);
    }

    indices
}
