//! Measuring the quality of multiple sequence alignments (MSAs) using various metrics.

#![expect(clippy::cast_precision_loss)]

use crate::{DistanceValue, Tree};

use super::Sequence;

/// Different quality metrics for evaluating MSAs.
pub enum QualityMetric {
    /// The mean fraction of gaps per sequence in the MSA.
    GapFraction(GapFraction),
    /// The fraction of mismatches between pairs of sequences in the MSA.
    MismatchFraction(MismatchFraction),
}

/// The mean of the fraction of gaps in the sequences of the MSA.
pub struct GapFraction(f64, f64);

impl GapFraction {
    /// Returns the mean gap fractions.
    pub const fn mean(&self) -> f64 {
        self.0
    }

    /// Returns the standard deviation of the gap fractions.
    pub const fn std_dev(&self) -> f64 {
        self.1
    }
}

impl<'a, Id, S, T, A, M> From<&'a Tree<Id, S, T, A, M>> for GapFraction
where
    S: Sequence,
    T: DistanceValue,
{
    fn from(msa_tree: &'a Tree<Id, S, T, A, M>) -> Self {
        let msa_width = msa_tree.items[0].1.as_ref().len();
        let gap_fractions = msa_tree
            .items
            .iter()
            .map(|(_, seq)| seq.gap_count() as f64 / msa_width as f64)
            .collect::<Vec<_>>();

        let total = gap_fractions.iter().sum::<f64>();
        let num_seqs = msa_tree.cardinality() as f64;
        let mean = total / num_seqs;

        let std_dev = (gap_fractions.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / num_seqs).sqrt();
        Self(mean, std_dev)
    }
}

/// The fraction of mismatches between pairs of sequences in the MSA.
pub struct MismatchFraction(f64, f64);

impl MismatchFraction {
    /// Returns the mean mismatch fraction.
    pub const fn mean(&self) -> f64 {
        self.0
    }

    /// Returns the standard deviation of the mismatch fraction.
    pub const fn std_dev(&self) -> f64 {
        self.1
    }
}

impl<'a, Id, S, T, A, M> From<&'a Tree<Id, S, T, A, M>> for MismatchFraction
where
    S: Sequence,
    T: DistanceValue,
{
    fn from(msa_tree: &'a Tree<Id, S, T, A, M>) -> Self {
        let mismatch_fractions = mm_inner(&msa_tree.items);
        let total: f64 = mismatch_fractions.iter().sum();
        let num_pairs = mismatch_fractions.len() as f64;
        let mean = total / num_pairs;
        let std_dev = (mismatch_fractions.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / num_pairs).sqrt();

        Self(mean, std_dev)
    }
}

/// A helper for calculating mismatch fractions between all pairs of sequences.
fn mm_inner<Id, I: AsRef<[u8]>>(seqs: &[(Id, I)]) -> Vec<f64> {
    seqs.iter()
        .enumerate()
        .flat_map(|(i, (_, s1))| {
            seqs.iter()
                .skip(i + 1)
                .map(move |(_, s2)| mm_single(s1.as_ref(), s2.as_ref()))
        })
        .collect()
}

/// A helper for calculating mismatch fraction of a single pair of sequences.
fn mm_single(s1: &[u8], s2: &[u8]) -> f64 {
    let num_mismatches = s1.iter().zip(s2.iter()).filter(|(a, b)| a != b).count();
    num_mismatches as f64 / s1.len() as f64
}
