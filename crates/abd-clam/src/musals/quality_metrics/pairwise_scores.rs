//! Scores each pairwise alignment in the MSA, applying a penalty for gaps and mismatches.

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
    musals::{CostMatrix, Sequence},
};

use super::{MsaQuality, mu_sigma_min_max, random_sample_indices};

/// The scores of pairwise alignments in the MSA.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct PairwiseScores {
    /// Mean
    mean: f64,
    /// Standard Deviation
    std_dev: f64,
    /// Minimum
    min: f64,
    /// Maximum
    max: f64,
}

impl MsaQuality for PairwiseScores {
    fn name(&self) -> String {
        "PairwiseScores".to_string()
    }

    fn short_name<'a>(&self) -> &'a str {
        "ps"
    }

    fn description(&self) -> String {
        "The mean score of all pairwise alignments in the MSA.".to_string()
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

    fn compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>, cost_matrix: &CostMatrix<T>, sample_size: Option<usize>) -> Self
    where
        S: Sequence,
        T: DistanceValue,
        M: Fn(&S, &S) -> T,
        Self: Sized,
    {
        let scorer = |(_, s1): &(Id, S), (_, s2): &(Id, S)| ps_inner(s1, s2, cost_matrix);
        let indices = random_sample_indices(msa_tree.cardinality(), sample_size);
        let pairwise_scores = apply_pairwise(&msa_tree.items, &indices, scorer).collect::<Vec<_>>();
        let (mean, std_dev, min, max) = mu_sigma_min_max(&pairwise_scores);
        Self { mean, std_dev, min, max }
    }

    fn par_compute<Id, S, T, A, M>(msa_tree: &Tree<Id, S, T, A, M>, cost_matrix: &CostMatrix<T>, sample_size: Option<usize>) -> Self
    where
        Id: Send + Sync,
        S: Sequence + Send + Sync,
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        M: Fn(&S, &S) -> T + Send + Sync,
        Self: Sized + Send + Sync,
    {
        let scorer = |(_, s1): &(Id, S), (_, s2): &(Id, S)| ps_inner(s1, s2, cost_matrix);
        let indices = random_sample_indices(msa_tree.cardinality(), sample_size);
        let pairwise_scores = par_apply_pairwise(&msa_tree.items, &indices, scorer).collect::<Vec<_>>();
        let (mean, std_dev, min, max) = mu_sigma_min_max(&pairwise_scores);
        Self { mean, std_dev, min, max }
    }
}

/// Applies a pairwise scorer to all pairs of sequences in the MSA.
pub fn apply_pairwise<S, F: Fn(&S, &S) -> f64>(sequences: &[S], indices: &[usize], scorer: F) -> impl Iterator<Item = f64> {
    indices
        .iter()
        .enumerate()
        .flat_map(move |(i, &s1)| indices.iter().skip(i + 1).map(move |&s2| (s1, s2)))
        .map(move |(i, j)| scorer(&sequences[i], &sequences[j]))
}

/// Parallel version of [`apply_pairwise`].
pub fn par_apply_pairwise<S: Send + Sync, F: Fn(&S, &S) -> f64 + Send + Sync>(
    sequences: &[S],
    indices: &[usize],
    scorer: F,
) -> impl ParallelIterator<Item = f64> {
    indices
        .par_iter()
        .enumerate()
        .flat_map(move |(i, &s1)| indices.par_iter().skip(i + 1).map(move |&s2| (s1, s2)))
        .map(move |(i, j)| scorer(&sequences[i], &sequences[j]))
}

/// Scores a single pairwise alignment in the MSA, applying a penalty for gaps and mismatches.
fn ps_inner<S: Sequence, T: DistanceValue>(s1: &S, s2: &S, cost_matrix: &CostMatrix<T>) -> f64 {
    let score = s1.as_ref().iter().zip(s2.as_ref().iter()).fold(T::zero(), |score, (&a, &b)| {
        if a == b {
            score
        } else if a == S::GAP || b == S::GAP {
            score + cost_matrix.gap_open_cost()
        } else {
            // a != b
            score + cost_matrix.sub_cost(a, b)
        }
    });

    score.to_f64().unwrap_or_else(|| unreachable!("DistanceValue to_f64 conversion failed"))
}
