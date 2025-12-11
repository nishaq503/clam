//! The fraction of mismatches between pairs of sequences in a multiple sequence alignment (MSA).

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
    musals::{CostMatrix, Sequence},
};

use super::{MsaQuality, mu_sigma_min_max};

/// The fraction of mismatches between pairs of sequences in the MSA.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct MismatchFraction {
    /// Mean
    mean: f64,
    /// Standard Deviation
    std_dev: f64,
    /// Minimum
    min: f64,
    /// Maximum
    max: f64,
}

impl MsaQuality for MismatchFraction {
    fn name(&self) -> String {
        "MismatchFraction".to_string()
    }

    fn short_name<'a>(&self) -> &'a str {
        "mf"
    }

    fn description(&self) -> String {
        "The mean fraction of mismatches between all pairs of sequences in the MSA.".to_string()
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn std_dev(&self) -> f64 {
        self.std_dev
    }

    fn min(&self) -> f64 {
        self.min
    }

    fn max(&self) -> f64 {
        self.max
    }

    fn compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>, _: &CostMatrix<T>) -> Self
    where
        S: Sequence,
        T: DistanceValue,
        M: Fn(&S, &S) -> T,
        Self: Sized,
    {
        let (mean, std_dev, min, max) = mu_sigma_min_max(mm_inner(&msa_tree.items));
        Self { mean, std_dev, min, max }
    }

    fn par_compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>, _: &CostMatrix<T>) -> Self
    where
        Id: Send + Sync,
        S: Sequence + Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Fn(&S, &S) -> T + Send + Sync,
        Self: Sized + Send + Sync,
    {
        let (mean, std_dev, min, max) = mu_sigma_min_max(par_mm_inner(&msa_tree.items));
        Self { mean, std_dev, min, max }
    }
}

/// A helper for calculating mismatch fractions between all pairs of sequences.
fn mm_inner<Id, I: AsRef<[u8]>>(seqs: &[(Id, I)]) -> Vec<f64> {
    seqs.iter()
        .enumerate()
        .flat_map(|(i, (_, s1))| seqs.iter().skip(i + 1).map(move |(_, s2)| mm_single(s1.as_ref(), s2.as_ref())))
        .collect()
}

/// A helper for calculating mismatch fraction of a single pair of sequences.
fn mm_single(s1: &[u8], s2: &[u8]) -> f64 {
    let num_mismatches = s1.iter().zip(s2.iter()).filter(|(a, b)| a != b).count();
    num_mismatches as f64 / s1.len() as f64
}

/// Parallel version of [`mm_inner`].
fn par_mm_inner<Id: Send + Sync, I: AsRef<[u8]> + Send + Sync>(seqs: &[(Id, I)]) -> Vec<f64> {
    seqs.par_iter()
        .enumerate()
        .flat_map(|(i, (_, s1))| seqs.par_iter().skip(i + 1).map(move |(_, s2)| mm_single(s1.as_ref(), s2.as_ref())))
        .collect()
}
